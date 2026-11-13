/*
 * This file is part of libplacebo.
 *
 * libplacebo is free software; you can redistribute it and/or
 * modify it under the terms of the GNU Lesser General Public
 * License as published by the Free Software Foundation; either
 * version 2.1 of the License, or (at your option) any later version.
 *
 * libplacebo is distributed in the hope that it will be useful,
 * but WITHOUT ANY WARRANTY; without even the implied warranty of
 * MERCHANTABILITY or FITNESS FOR A PARTICULAR PURPOSE.  See the
 * GNU Lesser General Public License for more details.
 *
 * You should have received a copy of the GNU Lesser General Public
 * License along with libplacebo.  If not, see <http://www.gnu.org/licenses/>.
 */

#include <math.h>
#include <stdlib.h>
#include <string.h>

#include "shaders.h"
#include "shaders/motion.h"
#include "dispatch.h"

#include <libplacebo/shaders/colorspace.h>
#include <libplacebo/shaders/motion.h>
#include <libplacebo/shaders/sampling.h>

const struct pl_motion_estimate_params pl_motion_estimate_default_params = {
    PL_MOTION_ESTIMATE_DEFAULTS
};

const struct pl_frame_interp_params pl_frame_interp_default_params = {
    PL_FRAME_INTERP_DEFAULTS
};

// One texture per level of an image pyramid, coarsest level last.
typedef PL_ARRAY(pl_tex) tex_pyramid_t;

// Persistent state and scratch space behind a PL_SHADER_OBJ_MOTION shader object.
struct sh_motion_obj {
    // --- Persistent across frames

    // Signatures of the frame pair the persistent state below describes, i.e.
    // the ones given to the most recent `pl_motion_estimate` call. `sig_b`
    // also identifies the frame `pyramid_prev` was built from.
    uint64_t sig_a, sig_b;

    // Luma pyramid of the *previous* call's frame `b`, reused as this call's
    // frame `a` pyramid. Only holds a usable pyramid if `pyramid_valid`.
    tex_pyramid_t pyramid_prev;
    bool pyramid_valid;

    // Block granular flow fields from each level of the image pyramid
    // searched by the most recent motion estimation invocation. Used to seed
    // the next motion estimation search.
    //
    // `dense_fwd` and `dense_bwd` are the per pixel refinements of
    // `fwd.elem[0]` and `bwd.elem[0]`, at the input resolution, and what
    // `pl_motion_get_flow` returns.
    // `valid` determines if `fwd` and `bwd` contain useful content.
    struct {
        tex_pyramid_t fwd;
        tex_pyramid_t bwd;
        pl_tex dense_fwd;
        pl_tex dense_bwd;
        enum pl_motion_direction valid; // directions actually computed
    } flow;

    // EXPERIMENT (stage B): the selected vectors, and those with the occlusion
    // estimate, of each direction; `b_mode` is whether B ran
    pl_tex b_sel[2], b_occ[2], b_shp[2], b_sub[2];
    int b_mode;

    // State of the splat, see `splat_prepare`: the vector and key prefix of
    // each pixel of both fields (`src`, rebuilt if `dirty`), the key buffers
    // of this and the next intermediate frame by turns (`frame` counts them),
    // and the pixels in front of others, as those holding `pair`
    struct {
        pl_buf src, key[2], mark, acc, qual;
        unsigned pair, frame;
        bool dirty;
    } splat;

    // Dimensions of `flow.fwd.elem[0]` (number of blocks to cover the full
    // resolution).
    int grid_w, grid_h;

    // Global motion of each direction, as an SSBO laid out as
    // `global_buf_data`. Not allocated until `input->estimate_global` is set.
    // `valid` determines if `fwd` and `bwd` hold the global motion of the
    // directions in `flow.valid`. `hist` is scratch for computation.
    struct {
        pl_buf fwd;
        pl_buf bwd;
        pl_buf hist;
        bool valid;
    } global;

    // LUTs for the search candidate ordering.
    // `cand_lut[0]` holds the fine search radius LUT and `cand_lut[1]` holds
    // the coarse one.
    pl_shader_obj cand_lut[2];

    // --- Valid only for the duration of one pl_motion_estimate call
    struct {
        tex_pyramid_t levels; // luma pyramid of this call's frame `b`
        tex_pyramid_t raw;    // per-level search output
        tex_pyramid_t filtered; // per-level search output after smoothing
    } scratch;

    struct pl_motion_estimate_params params;
    int w, h;
    pl_tex raw_dbg[2];
    pl_tex fine_raw[4], fine[2][4];
};

// Binds `tex` for use with `texelFetch`.
static ident_t bind_tex(pl_shader sh, const char *name, pl_tex tex)
{
    return sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name = name,
            .type = PL_DESC_SAMPLED_TEX,
        },
        .binding = {
            .object      = tex,
            .sample_mode = PL_TEX_SAMPLE_NEAREST,
        },
    });
}

// Binds `tex` as a write-only storage image.
static ident_t bind_img(pl_shader sh, const char *name, pl_tex tex)
{
    return sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = name,
            .type   = PL_DESC_STORAGE_IMG,
            .access = PL_DESC_ACCESS_WRITEONLY,
        },
        .binding.object = tex,
    });
}

// Binds `buf` as a storage buffer holding `vars`.
static void bind_buf(pl_shader sh, const char *name, pl_buf buf,
                     enum pl_desc_access access,
                     const struct pl_buffer_var *vars, int num_vars)
{
    sh_desc(sh, (struct pl_shader_desc) {
        .desc = {
            .name   = name,
            .type   = PL_DESC_BUF_STORAGE,
            .access = access,
        },
        .binding.object  = buf,
        .buffer_vars     = (struct pl_buffer_var *) vars,
        .num_buffer_vars = num_vars,
    });
}

// Don't reduce image pyramid to a size that is too small to recover motion.
#define MIN_COARSE_BLOCKS 3

// "Block" will be used to refer to a block of pixels that we group together
// for the motion search.
// "Tile" will be used to refer to a set of blocks that one workgroup operates
// on.
// `MOTION_BPGX` and `MOTION_BPGY` define the tile dimensions.
//
// TODO: benchmark the impacts of varying shape and size of tile to configure
// texel reuse and memory access patterns. Also needs to be revisited when we
// start the motion search at different locations for each block depending on
// the temporal state.
#define MOTION_BPGX 2
#define MOTION_BPGY 5

// Side length of the window of pixels that `motion_refine_pass` matches
// around every pixel.
#define MOTION_REFINE_WINDOW 3

// Margin by which a vector of a neighbouring block must match a pixel better
// than the vector of its own block for `motion_refine_pass` to take it, as a
// mean squared error per pixel in 8 bit code values.
#define MOTION_REFINE_MARGIN 1.0f

// Mismatch, relative to the texture energy of the window, at which
// `motion_refine_pass` halves the confidence of a pixel's vector.
#define MOTION_REFINE_OCCLUSION 0.3f

// Added to the texture energy that the mismatch is relative to, to keep it
// finite in flat areas, as a mean squared error per pixel in 8 bit code values.
#define MOTION_REFINE_FLAT 1.0f

// Brightness difference at which a visible pixel no longer supports the vector
// of a hidden one, as a Gaussian sigma in units of the luma textures.
#define MOTION_FILL_SIGMA 0.05f

// Scale of the mismatch with the other frame in the splat key, and the value
// of pixels hidden there, which cannot be matched.
#define MOTION_SPLAT_COST 1024.0f
#define MOTION_SPLAT_COST_HIDDEN 102u

// Brightness difference at which the splat still mixes in the sample of a
// frame its winner is hidden in, as a Gaussian sigma of the square root of the
// linear luminance.
#define MOTION_SPLAT_SIMILAR 0.2f

// Total number of bins considered when creating the histogram of global
// motion. NOTE: if the median motion is outside of [-`MOTION_GLOBAL_BINS/2`,
// `MOTION_GLOBAL_BINS-1`], the global motion is clamped.
#define MOTION_GLOBAL_BINS 256

// The shader that creates the histogram of motion uses a workgroup with the
// dimensions (`MOTION_GLOBAL_BINS`, `MOTION_GLOBAL_BINS`)
#define MOTION_GLOBAL_WORKGROUP_DIM 16
pl_static_assert(MOTION_GLOBAL_WORKGROUP_DIM *MOTION_GLOBAL_WORKGROUP_DIM ==
                 MOTION_GLOBAL_BINS);

// The different seed candidates that the motion search will consider, in the
// order that breaks ties.
enum motion_seed_kind {
    MOTION_SEED_ZERO,
    MOTION_SEED_PARENT,
    MOTION_SEED_PARENT_X, // the parent's horizontal neighbour nearest the block
    MOTION_SEED_PARENT_Y, // the parent's vertical neighbour nearest the block
    MOTION_SEED_PRIOR,
    MOTION_SEED_NUM,
};

// See `motion_seeds_plan`.
struct motion_seeds {
    int rank[MOTION_SEED_NUM];
    int num;
};

// Plans which of the `motion_seed_kind` a given level of the motion search
// pyramid will consider. `rank[k]` == -1 when candidate `k` will not be
// considered, `rank[k]` == index of candidate `k` in ranking of seeds that
// will be considered.
static struct motion_seeds motion_seeds_plan(bool has_parent, bool has_prior)
{
    struct motion_seeds seeds = {0};
    for (int i = 0; i < MOTION_SEED_NUM; i++)
        seeds.rank[i] = -1;

    seeds.rank[MOTION_SEED_ZERO] = seeds.num++;

    if (has_parent) {
        seeds.rank[MOTION_SEED_PARENT] = seeds.num++;
        seeds.rank[MOTION_SEED_PARENT_X] = seeds.num++;
        seeds.rank[MOTION_SEED_PARENT_Y] = seeds.num++;
    }

    if (has_prior)
        seeds.rank[MOTION_SEED_PRIOR] = seeds.num++;

    return seeds;
}

// Constants that will be used for motion estimation. Varies depending on
// device capabilities and user selected parameters.
struct motion_geometry {
    int block_size;    // dimensions of square block size
    int radius;        // search radius, in pixels of the current image pyramid
                       // level
    int bpgx, bpgy;    // dimensions of tile
    int threads_block; // threads per block in the tile
    bool subpixel;     // refine the vectors to a fraction of a pixel

    // The seed candidates every block of this level scores before searching.
    struct motion_seeds seeds;

    // "Candidate" will be used to refer to a motion vector (a vector of (dx, dy)
    // with 0 <= dx,dy <= radius) that is considered for a given block of pixels.
    // The ordering of the `(2 * radius + 1)^2` motion candidates in the
    // search radius, for tie breaking order.
    // For i in [0, `(2 * radius + 1)^2`), i is the ordering for tie breaking,
    // and (cand[2*i], cand[2*i+1]) represents (dx, dy) of the corresponding
    // motion search candidate.
    const int *cand;
};

// The size and stride of the various contents of shared memory that a motion
// search shader will use.
// "Window" will be used to refer to all of the pixels that are possible
// motion vector candidates for a given block.
// TODO: revisit strides for bank conflicts.
struct motion_shmem_layout {
    int win;          // dimensions of square window
    int win_stride;   // row stride inside a search window
    int block_stride; // stride between different windows of a tile
    size_t size;
};

// TODO: consider if we can pack multiple pixels into a single shared memory bank.
static struct motion_shmem_layout
motion_compute_shmem_layout(int block_size, int radius, int bpgx, int bpgy,
                            int nseed, bool subpixel)
{
    const int win = block_size + 2 * radius;
    const int win_stride = win | 1;
    const int block_stride = (win * win_stride) | 1;
    const int blocks = bpgx * bpgy;

    // When `nseed` == 1, that is the case when the only seed candidate is
    // the zero vector. We do not need to score any candidates in this case.
    const size_t score = nseed > 1 ? (size_t) blocks * nseed : 0;

    const size_t words =
        (size_t) blocks * PL_SQUARE(block_size) // reference tile
        + (size_t) blocks * block_stride        // search windows
        + 3 * (size_t) blocks                   // cost, rank and cost sum
        + 2 * (size_t) blocks                   // seed
        + (subpixel ? 4 : 0) * (size_t) blocks  // cost around the winner
        + score;                                // seed candidate scores

    return (struct motion_shmem_layout) {
        .win = win,
        .win_stride = win_stride,
        .block_stride = block_stride,
        .size = words * sizeof(uint32_t),
    };
}

// Fills the `out` array with `(2 * radius + 1)^2` many candidate displacements
// of the search window. Candidate i will always represent the displacement at
// (out[2*i], out[2*i + 1]), and the index i will be used to make the tie
// breaking order deterministic and favoring smaller displacements.
//
// We measure the magnitude of displacements using Chebyshev distance.
static void cand_fill(int radius, int *out)
{
    int num = 0;
    for (int ring = 0; ring <= radius; ring++) {
        for (int dy = -ring; dy <= ring; dy++) {
            for (int dx = -ring; dx <= ring; dx++) {
                const int adx = abs(dx), ady = abs(dy);
                if (PL_MAX(adx, ady) != ring)
                    continue; // everything closer has already been written

                out[2 * num + 0] = dx;
                out[2 * num + 1] = dy;
                num++;
            }
        }
    }

    pl_assert(num == PL_SQUARE(2 * radius + 1)); // correct number of elements
    pl_assert(!out[0] && !out[1]); // first element of `out` is (0,0)
}

// Selects an appropriate shader geometry (grid and workgroup dimensions) for
// the motion estimation. Iteratively reduces the tile size that each workgroup
// will process until a size is found that fits in shared memory.
static bool motion_plan(pl_gpu gpu, void *alloc, int block_size, int radius,
                        struct motion_seeds seeds, bool subpixel, int grid_w,
                        int grid_h, struct motion_geometry *out)
{
    const int ncand = PL_SQUARE(2 * radius + 1);
    const int max_group = gpu->glsl.max_group_threads;
    const int max_x =
        gpu->glsl.max_group_size[0]; // TODO: check if this is limiting factor
    const int max_threads = PL_MIN(max_group, max_x);

    // This should not occur for any reasonable radius choice.
    const int threads_block = PL_MIN(ncand, max_threads);

    // This should not occur for any reasonable resolution of image.
    int bpgx = PL_MIN(MOTION_BPGX, grid_w);
    int bpgy = PL_MIN(MOTION_BPGY, grid_h);

    while (bpgx > 1 || bpgy > 1) {
        const struct motion_shmem_layout shmem = motion_compute_shmem_layout(
            block_size, radius, bpgx, bpgy, seeds.num, subpixel);

        if (bpgx * bpgy * threads_block <= max_threads &&
            shmem.size <= gpu->glsl.max_shmem_size)
            break;

        if (bpgx >= bpgy)
            bpgx--;
        else
            bpgy--;
    }

    const struct motion_shmem_layout shmem = motion_compute_shmem_layout(
        block_size, radius, bpgx, bpgy, seeds.num, subpixel);

    if (bpgx * bpgy * threads_block > max_threads) {
        PL_ERR(gpu,
               "pl_motion_estimate: search radius %d needs %d threads per work "
               "group, but this GPU supports at most %d!",
               radius, bpgx * bpgy * threads_block, max_threads);
        return false;
    }

    if (shmem.size > gpu->glsl.max_shmem_size) {
        PL_ERR(gpu,
               "pl_motion_estimate: block size %d with search radius %d needs "
               "%zu bytes of shared memory per work group, but this GPU "
               "provides only %zu!",
               block_size, radius, shmem.size, gpu->glsl.max_shmem_size);
        return false;
    }

    int *cand = pl_calloc(alloc, 2 * ncand, sizeof(*cand));
    cand_fill(radius, cand);

    *out = (struct motion_geometry) {
        .block_size = block_size,
        .radius = radius,
        .bpgx = bpgx,
        .bpgy = bpgy,
        .threads_block = threads_block,
        .subpixel = subpixel,
        .seeds = seeds,
        .cand = cand,
    };

    return true;
}

// Destroys every level of `pyramid`, along with the array holding them.
static void pyramid_destroy(pl_gpu gpu, tex_pyramid_t *pyramid)
{
    for (int i = 0; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);
    pl_free_ptr(&pyramid->elem);
    pyramid->num = 0;
}

// Number of levels before degenerating into a single pixel.
static int levels_supported(int w, int h)
{
    int levels = 1;
    while (w > 1 || h > 1) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        levels++;
    }
    return levels;
}

// Number of levels to use in the absence of a user provided value.
static int levels_auto(int w, int h, int block_size)
{
    const int min_size = MIN_COARSE_BLOCKS * block_size;
    int levels = 1;
    while (PL_MIN((w + 1) / 2, (h + 1) / 2) >= min_size) {
        w = (w + 1) / 2;
        h = (h + 1) / 2;
        levels++;
    }
    return levels;
}

static int pyramid_num_levels(int w, int h,
                              const struct pl_motion_estimate_params *params)
{
    if (!params->pyramid_levels)
        return levels_auto(w, h, params->block_size);

    return PL_MIN(params->pyramid_levels, levels_supported(w, h));
}

// Allocate or resize `pyramid` to hold exactly `levels` textures, level `i`
// being the input resolution halved `i` times (rounding up).
static bool pyramid_recreate(pl_gpu gpu, void *alloc, tex_pyramid_t *pyramid,
                             int w, int h, int levels, pl_fmt fmt)
{
    // If size of the frame reduced or block size increased since previous motion
    // estimate, destroy the pyramid allocations that would be unused.
    for (int i = levels; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);

    // If size of the frame increased since previous motion estimate, or this
    // is a cold start, prepare the indices to be allocated by `pl_tex_recreate`.
    PL_ARRAY_RESIZE(alloc, *pyramid, levels);
    for (int i = pyramid->num; i < levels; i++)
        pyramid->elem[i] = NULL;
    pyramid->num = levels;

    for (int i = 0; i < levels; i++) {
        bool ok = pl_tex_recreate(
            gpu, &pyramid->elem[i],
            pl_tex_params(.w = w, .h = h, .format = fmt, .sampleable = true,
                          .renderable = true,
                          .storable = fmt->caps & PL_FMT_CAP_STORABLE,
                          .debug_tag = PL_DEBUG_TAG, ));

        if (!ok) {
            PL_ERR(gpu,
                   "pl_motion_estimate: failed creating %dx%d pyramid "
                   "level!",
                   w, h);
            return false;
        }

        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    return true;
}

// Format for motion vector fields, holding `vec3(mv.x, mv.y, confidence)`,
// with an empty fourth element for padding. Prefer 16 bit floats but fall back
// to 32 bits.
static pl_fmt flow_fmt(pl_gpu gpu)
{
    static const enum pl_fmt_caps caps = PL_FMT_CAP_STORABLE |
                                         PL_FMT_CAP_LINEAR;

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 16, 0, caps);
    if (fmt)
        return fmt;

    return pl_find_fmt(gpu, PL_FMT_FLOAT, 4, 32, 0, caps);
}

// Similar to `pyramid_recreate`, except does not invalidate the contents of
// `pyramid` when the dimensions and levels of the flow pyramid has not changed
// since the previous call.
static bool flow_pyramid_recreate(pl_gpu gpu, void *alloc,
                                  tex_pyramid_t *pyramid, int w, int h,
                                  int levels, int block_size, pl_fmt fmt)
{
    bool reusable = pyramid->num == levels;
    for (int i = 0, lw = w, lh = h; reusable && i < levels; i++) {
        pl_tex tex = pyramid->elem[i];
        reusable = tex && tex->params.format == fmt &&
                   tex->params.w == PL_DIV_UP(lw, block_size) &&
                   tex->params.h == PL_DIV_UP(lh, block_size);
        lw = (lw + 1) / 2;
        lh = (lh + 1) / 2;
    }

    if (reusable)
        return true;

    // Free levels that are now unused.
    for (int i = levels; i < pyramid->num; i++)
        pl_tex_destroy(gpu, &pyramid->elem[i]);

    // Clear levels that will be reused.
    PL_ARRAY_RESIZE(alloc, *pyramid, levels);
    for (int i = pyramid->num; i < levels; i++)
        pyramid->elem[i] = NULL;
    pyramid->num = levels;

    for (int i = 0; i < levels; i++) {
        // Use PL_DIV_UP to account for trailing partial blocks.
        const int grid_w = PL_DIV_UP(w, block_size);
        const int grid_h = PL_DIV_UP(h, block_size);

        bool ok = pl_tex_recreate(
            gpu, &pyramid->elem[i],
            pl_tex_params(.w = grid_w, .h = grid_h, .format = fmt,
                          .sampleable = true, .storable = true, .host_readable = true,
                          .debug_tag = PL_DEBUG_TAG, ));

        if (!ok) {
            PL_ERR(gpu, "pl_motion_estimate: failed creating %dx%d flow field!",
                   grid_w, grid_h);
            return false;
        }

        w = (w + 1) / 2;
        h = (h + 1) / 2;
    }

    return true;
}

// Allocate or resize `*tex` to hold a flow field of the input resolution.
static bool dense_flow_recreate(pl_gpu gpu, pl_tex *tex, int w, int h,
                                pl_fmt fmt)
{
    bool ok = pl_tex_recreate(gpu, tex, pl_tex_params(
        .w = w, .h = h, .format = fmt, .sampleable = true, .storable = true,
        .host_writable = true, .host_readable = true, // EXPERIMENT
        .debug_tag = PL_DEBUG_TAG, ));

    if (!ok)
        PL_ERR(gpu, "pl_motion_estimate: failed creating %dx%d flow field!", w, h);
    return ok;
}

// Downscale `src` by one pyramid level into `dst` using the Burt-Adelson REDUCE
// filter, i.e. convolution with the separable binomial kernel with a weighting
// of [1 4 6 4 1]/16. `dst` must measure exactly `src` halved in each dimension,
// rounding up.
static bool pyramid_reduce(pl_dispatch dispatch, pl_tex src, pl_tex dst)
{
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "pyramid REDUCE %dx%d -> %dx%d", src->params.w,
                 src->params.h, dst->params.w, dst->params.h);
    sh->output = PL_SHADER_SIG_COLOR;

    // Align the output texel (i,j) on the center of source texel (2i,2j).
    const pl_rect2df rect = {
        .x0 = -0.5f,
        .y0 = -0.5f,
        .x1 = 2.0f * dst->params.w - 0.5f,
        .y1 = 2.0f * dst->params.h - 0.5f,
    };

    ident_t pos, pt;
    ident_t tex = sh_bind(sh, src, PL_TEX_ADDRESS_MIRROR, PL_TEX_SAMPLE_LINEAR,
                          "src", &rect, &pos, &pt);
    if (!tex) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    // clang-format off
#pragma GLSL /* pyramid_reduce */                                               \
    vec4 color;                                                                 \
    {                                                                           \
        vec2 c = $pos;                                                          \
        vec2 d = 1.2 * $pt;                                                     \
        float sum  = 36.0 * textureLod($tex, c, 0.0).r;                         \
        sum += 30.0 * textureLod($tex, c - vec2(d.x, 0.0), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c + vec2(d.x, 0.0), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c - vec2(0.0, d.y), 0.0).r;              \
        sum += 30.0 * textureLod($tex, c + vec2(0.0, d.y), 0.0).r;              \
        sum += 25.0 * textureLod($tex, c - d, 0.0).r;                           \
        sum += 25.0 * textureLod($tex, c + d, 0.0).r;                           \
        sum += 25.0 * textureLod($tex, c - vec2(d.x, -d.y), 0.0).r;             \
        sum += 25.0 * textureLod($tex, c + vec2(d.x, -d.y), 0.0).r;             \
        color = vec4(sum * (1.0 / 256.0), 0.0, 0.0, 1.0);                       \
    }
    // clang-format on

    return pl_dispatch_finish(dispatch, pl_dispatch_params(
        .shader = &sh,
        .target = dst,
    ));
}

// pyramid->elem[0] must contain the luma plane of the corresponding frame.
static bool pyramid_build(pl_gpu gpu, pl_dispatch dispatch,
                          const tex_pyramid_t *pyramid)
{
    for (int i = 1; i < pyramid->num; i++) {
        if (!pyramid_reduce(dispatch, pyramid->elem[i-1], pyramid->elem[i])) {
            PL_ERR(gpu, "pl_motion_estimate: failed reducing pyramid level "
                   "%d!", i);
            return false;
        }
    }

    return true;
}

static void motion_uninit(pl_gpu gpu, void *ptr)
{
    struct sh_motion_obj *obj = ptr;
    pyramid_destroy(gpu, &obj->pyramid_prev);
    pyramid_destroy(gpu, &obj->scratch.levels);
    pyramid_destroy(gpu, &obj->scratch.raw);
    pyramid_destroy(gpu, &obj->scratch.filtered);
    pyramid_destroy(gpu, &obj->flow.fwd);
    pyramid_destroy(gpu, &obj->flow.bwd);
    pl_tex_destroy(gpu, &obj->flow.dense_fwd);
    pl_tex_destroy(gpu, &obj->flow.dense_bwd);
    for (int i = 0; i < 2; i++) {
        pl_tex_destroy(gpu, &obj->b_sel[i]);
        pl_tex_destroy(gpu, &obj->b_sub[i]);
        pl_tex_destroy(gpu, &obj->b_occ[i]);
        pl_tex_destroy(gpu, &obj->b_shp[i]);
    }
    pl_buf_destroy(gpu, &obj->splat.src);
    pl_buf_destroy(gpu, &obj->splat.key[0]);
    pl_buf_destroy(gpu, &obj->splat.key[1]);
    pl_buf_destroy(gpu, &obj->splat.mark);
    pl_buf_destroy(gpu, &obj->splat.acc);
    pl_buf_destroy(gpu, &obj->splat.qual);
    pl_tex_destroy(gpu, &obj->raw_dbg[0]);
    pl_tex_destroy(gpu, &obj->raw_dbg[1]);
    for (int i = 0; i < 4; i++) {
        pl_tex_destroy(gpu, &obj->fine_raw[i]);
        pl_tex_destroy(gpu, &obj->fine[0][i]);
        pl_tex_destroy(gpu, &obj->fine[1][i]);
    }
    pl_buf_destroy(gpu, &obj->global.fwd);
    pl_buf_destroy(gpu, &obj->global.bwd);
    pl_buf_destroy(gpu, &obj->global.hist);
    for (int i = 0; i < PL_ARRAY_SIZE(obj->cand_lut); i++)
        pl_shader_obj_destroy(&obj->cand_lut[i]);
    memset(obj, 0, sizeof(*obj));
}

// Drop everything that only makes sense in the context of a particular frame
// sequence. Invalidates but does not release allocated GPU buffers.
static void invalidate_temporal(pl_gpu gpu, struct sh_motion_obj *obj,
                                const char *reason)
{
    PL_DEBUG(gpu, "pl_motion: discarding temporal state: %s", reason);

    obj->sig_a = obj->sig_b = 0;
    obj->flow.valid = 0;
    obj->global.valid = false;
    obj->pyramid_valid = false;

    pl_buf_destroy(gpu, &obj->global.fwd);
    pl_buf_destroy(gpu, &obj->global.bwd);
    pl_buf_destroy(gpu, &obj->global.hist);
}

int pl_motion_num_blocks(int w, int h,
                         const struct pl_motion_estimate_params *params)
{
    const int bs = params->block_size;
    if (w <= 0 || h <= 0 || bs <= 0)
        return 0;

    // Must match the grid computed by `pl_motion_estimate`.
    return PL_DIV_UP(w, bs) * PL_DIV_UP(h, bs);
}

void pl_motion_reset(pl_shader_obj state)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj)
        return;

    invalidate_temporal(state->gpu, obj, "reset by application");
}

pl_tex pl_motion_get_flow(pl_shader_obj state, enum pl_motion_direction dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || !(obj->flow.valid & dir))
        return NULL;

    switch (dir) {
    case PL_MOTION_FORWARD:
        return obj->flow.dense_fwd;
    case PL_MOTION_BACKWARD:
        return obj->flow.dense_bwd;
    default:
        break;
    }

    return NULL; // `dir` cannot name `PL_MOTION_BOTH`, we can only return
                 // one flow field.
}

bool pl_motion_debug_set_flow(pl_shader_obj state, enum pl_motion_direction dir,
                              const void *data)
{
    pl_tex tex = pl_motion_get_flow(state, dir);
    if (!tex)
        return false;
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    obj->splat.dirty = true;
    return pl_tex_upload(state->gpu, pl_tex_transfer_params(.tex = tex, .ptr = (void *) data));
}

static bool motion_b_post(pl_gpu gpu, pl_dispatch dispatch,
                          struct sh_motion_obj *obj, pl_tex luma_a, pl_tex luma_b);

// EXPERIMENT: the per pixel selection of `dir` (0 fwd, 1 bwd), before the
// occlusion and fill passes
pl_tex pl_motion_debug_get_sel(pl_shader_obj state, int dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    return obj ? obj->b_sel[dir] : NULL;
}

// EXPERIMENT: re-run the occlusion and fill passes on the (modified) selection
bool pl_motion_debug_redo_b(pl_shader_obj state, pl_dispatch dp)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || !obj->b_mode)
        return false;
    obj->splat.dirty = true;
    // after the estimate, frame a's luma is in the scratch pyramid
    return motion_b_post(state->gpu, dp, obj, obj->scratch.levels.elem[0],
                         obj->pyramid_prev.elem[0]);
}

// EXPERIMENT (stage B): tunables read from the environment
static float b_envf(const char *name, float def)
{
    const char *v = getenv(name);
    return v ? atof(v) : def;
}

static int dc_env(const char *name, int def)
{
    const char *v = getenv(name);
    return v ? atoi(v) : def;
}

static ident_t bind_tex_linear(pl_shader sh, const char *name, pl_tex tex)
{
    return sh_desc(sh, (struct pl_shader_desc) {
        .desc = { .name = name, .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = tex, .sample_mode = PL_TEX_SAMPLE_LINEAR,
                     .address_mode = PL_TEX_ADDRESS_CLAMP },
    });
}


pl_tex pl_motion_debug_fine(pl_shader_obj state, int dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    int n = getenv("FINE") ? atoi(getenv("FINE")) : 0;
    return obj && n > 0 ? obj->fine[dir][n] : NULL;
}

pl_tex pl_motion_debug_raw(pl_shader_obj state, int dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    return obj ? obj->raw_dbg[dir] : NULL;
}

ident_t sh_motion_flow(pl_shader sh, pl_shader_obj state,
                       enum pl_motion_direction dir)
{
    pl_tex flow = pl_motion_get_flow(state, dir);
    if (!flow)
        return NULL_IDENT;

    return bind_tex(sh, "motion_flow", flow);
}

pl_buf pl_motion_get_global_buf(pl_shader_obj state,
                                enum pl_motion_direction dir)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || !obj->global.valid || !(obj->flow.valid & dir))
        return NULL;

    switch (dir) {
    case PL_MOTION_FORWARD:
        return obj->global.fwd;
    case PL_MOTION_BACKWARD:
        return obj->global.bwd;
    default:
        break;
    }

    return NULL; // `dir` cannot name `PL_MOTION_BOTH`, we can only return
                 // one global motion vector.
}

struct global_buf_data {
    float global_mv[2];
    float global_confidence;
};

const struct pl_buffer_var
    pl_motion_global_buf_vars[PL_MOTION_BACKWARD + 1]
                             [PL_MOTION_GLOBAL_BUF_NUM_VARS] = {
#define VAR(field, suffix, dim) {                                               \
    .var = {                                                                    \
        .name = #field "_" #suffix,                                             \
        .type = PL_VAR_FLOAT,                                                   \
        .dim_v = dim,                                                           \
        .dim_m = 1,                                                             \
        .dim_a = 1,                                                             \
    },                                                                          \
    .layout = {                                                                 \
        .offset = offsetof(struct global_buf_data, field),                      \
        .size   = sizeof(((struct global_buf_data *) NULL)->field),             \
        .stride = sizeof(((struct global_buf_data *) NULL)->field),             \
    },                                                                          \
}
    // NOTE: index 0 is empty.
    [PL_MOTION_FORWARD] = {
        VAR(global_mv, fwd, 2),
        VAR(global_confidence, fwd, 1),
    },
    [PL_MOTION_BACKWARD] = {
        VAR(global_mv, bwd, 2),
        VAR(global_confidence, bwd, 1),
    },
#undef VAR
};

ident_t sh_motion_global(pl_shader sh, pl_shader_obj state,
                         enum pl_motion_direction dir)
{
    pl_buf global = pl_motion_get_global_buf(state, dir);
    if (!global)
        return NULL_IDENT;

    pl_assert(global->params.size >= sizeof(struct global_buf_data));

    bind_buf(sh, "global_motion_desc", global, PL_DESC_ACCESS_READONLY,
             pl_motion_global_buf_vars[dir], PL_MOTION_GLOBAL_BUF_NUM_VARS);

    ident_t gm = sh_fresh(sh, "global_motion");
    // clang-format off
#pragma GLSLP /* sh_motion_global */                                            \
    @if (dir == PL_MOTION_FORWARD)                                              \
    #define $gm vec3(global_mv_fwd, global_confidence_fwd)                      \
    @else                                                                       \
    #define $gm vec3(global_mv_bwd, global_confidence_bwd)                      \

    // clang-format on
    return gm;
}

bool pl_motion_get_global(pl_shader_obj state, enum pl_motion_direction dir,
                          float *out_x, float *out_y, float *out_confidence)
{
    pl_buf global = pl_motion_get_global_buf(state, dir);
    if (!global)
        return false;

    struct global_buf_data data;
    if (!pl_buf_read(state->gpu, global, 0, &data, sizeof(data)))
        return false;

    if (out_x)
        *out_x = data.global_mv[0];
    if (out_y)
        *out_y = data.global_mv[1];
    if (out_confidence)
        *out_confidence = data.global_confidence;
    return true;
}

static bool params_eq(const struct pl_motion_estimate_params *a,
                      const struct pl_motion_estimate_params *b)
{
    return a->block_size == b->block_size &&
           a->pyramid_levels == b->pyramid_levels &&
           a->search_radius_coarse == b->search_radius_coarse &&
           a->search_radius_fine == b->search_radius_fine;
}

static bool validate(pl_gpu gpu, const struct pl_motion_estimate_input *input,
                     const struct pl_motion_estimate_params *params)
{
    if (!gpu->glsl.compute) {
        PL_ERR(gpu, "pl_motion_estimate: requires compute shader support!");
        return false;
    }

    if (!input->a.tex || !input->b.tex) {
        PL_ERR(gpu, "pl_motion_estimate: missing input texture!");
        return false;
    }

    const struct pl_tex_params *a = &input->a.tex->params;
    const struct pl_tex_params *b = &input->b.tex->params;
    if (a->w != b->w || a->h != b->h) {
        PL_ERR(gpu, "pl_motion_estimate: input dimensions differ (%dx%d vs "
               "%dx%d)!", a->w, a->h, b->w, b->h);
        return false;
    }

    if (!a->sampleable || !b->sampleable) {
        PL_ERR(gpu, "pl_motion_estimate: input textures must be sampleable!");
        return false;
    }

    if (input->a.sig == input->b.sig) {
        PL_ERR(gpu, "pl_motion_estimate: a.sig and b.sig must identify "
               "distinct frames (both 0x%llx)!",
               (unsigned long long) input->a.sig);
        return false;
    }

    if (input->direction & ~(unsigned) PL_MOTION_BOTH || !input->direction) {
        PL_ERR(gpu, "pl_motion_estimate: invalid direction %d!",
               (int) input->direction);
        return false;
    }

    if (params->block_size < 1) {
        PL_ERR(gpu, "pl_motion_estimate: block_size must be positive!");
        return false;
    }

    if (params->block_size > 64) {
        PL_ERR(gpu, "pl_motion_estimate: block_size must be <= 64!");
        return false;
    }

    if (params->pyramid_levels < 0) {
        PL_ERR(gpu, "pl_motion_estimate: pyramid_levels must be non-negative!");
        return false;
    }

    if (params->search_radius_coarse < 0 || params->search_radius_fine < 0) {
        PL_ERR(gpu, "pl_motion_estimate: search radii must be non-negative!");
        return false;
    }

    if (params->search_radius_coarse > 64 || params->search_radius_fine > 64) {
        PL_ERR(gpu, "pl_motion_estimate: search radii must be <= 64!");
        return false;
    }

    return true;
}

// Convert an input frame to luma.
static bool convert_to_luma(pl_dispatch dispatch,
                            const struct pl_motion_estimate_frame *frame,
                            pl_tex out)
{
    struct pl_color_repr repr = frame->repr;
    struct pl_color_space color = frame->color;
    pl_color_space_infer(&color);

    pl_shader sh = pl_dispatch_begin(dispatch);
    if (!pl_shader_sample_direct(sh, pl_sample_src( .tex = frame->tex ))) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    // Decode frame to RGB if required. `pl_shader_decode_color_ex` also ensures
    // `PL_ALPHA_INDEPENDENT`, which is required by `pl_shader_extract_features`.
    pl_shader_decode_color_ex(sh, pl_color_decode_args( .repr = &repr ));

    // Convert to luma (IPT intensity). `pl_shader_extract_features` linearizes
    // for us.
    pl_shader_extract_features(sh, color);

    return pl_dispatch_finish(dispatch, pl_dispatch_params(
        .shader = &sh,
        .target = out,
    ));
}

static void cand_lut_fill(void *data, const struct sh_lut_params *params)
{
    const struct motion_geometry *geom = params->priv;
    memcpy(data, geom->cand, 2 * (size_t) params->width * sizeof(int));
}

// All of the parameters for a motion search shader for one level of the image
// pyramid.
struct motion_search {
    pl_tex ref;                         // reference frame luma
    pl_tex tgt;                         // target frame luma
    pl_tex parent; // flow field of the previous motion search image pyramid
                   // level, or NULL at the coarsest level
    pl_tex prior;  // flow field produced at the same image pyramid level by
                   // the motion search invocation on the previous frame pair,
                   // or NULL when there is none available
    pl_tex out_mv;                      // block granular flow field
    const struct motion_geometry *geom; // compile time constants
    int parent_scale;                   // EXPERIMENT: 0 = 2
    pl_shader_obj *cand_lut;
};

// Dispatches the motion search shader for every block of `p->out_mv`.
//
// Uses the sum of squared distances (SSD) to compute the most likely
// motion vector that describes the motion of a block of pixels from one
// frame to the next.
//
// Also stores the distinctiveness of that match, i.e. how much lower the SSD
// of the winning candidate is than the average SSD of the search window:
//
//     q = 1 - (SSD_min + noise) / (SSD_mean + noise)
//
// This is 0 for a block that matches every candidate equally well, e.g. for
// lack of texture, and independent of the contrast of the source.
static bool motion_search_pass(pl_gpu gpu, pl_dispatch dispatch,
                               const struct motion_search *p)
{
    const struct motion_geometry *geom = p->geom;
    const int bs      = geom->block_size;
    const int r       = geom->radius;
    const int bpgx    = geom->bpgx, bpgy = geom->bpgy;
    const int bpg     = bpgx * bpgy;
    const int tpb     = geom->threads_block;
    const int threads = bpg * tpb;
    const int ncand   = PL_SQUARE(2 * r + 1);
    const int nseed = geom->seeds.num;
    const bool subpixel = geom->subpixel;

    // Drop as many low bits of each SSD as needed for the sum of the SSD of
    // every candidate to fit into a uint. Zero as long as (2r+1) * bs <= 257.
    const uint64_t max_ssd = (uint64_t) PL_SQUARE(bs) * 255 * 255;
    int cost_shift = 0;
    while ((ncand * max_ssd) >> cost_shift > UINT32_MAX)
        cost_shift++;

    // SSD below which a match is indistinguishable from quantization noise,
    // i.e. a variance of 1/4 per pixel (the difference of two luma values
    // rounded to 8 bits has a variance of 1/6).
    const float cost_noise = PL_SQUARE(bs) / 4.0f / (1 << cost_shift);

    // Total size of tile.
    const int ref_w = bpgx * bs, ref_h = bpgy * bs;

    const struct motion_shmem_layout shmem =
        motion_compute_shmem_layout(bs, r, bpgx, bpgy, nseed, subpixel);
    const int win = shmem.win;
    const int win_stride = shmem.win_stride;
    const int block_stride = shmem.block_stride;

    const int tex_w = p->ref->params.w, tex_h = p->ref->params.h;
    const int grid_w = p->out_mv->params.w, grid_h = p->out_mv->params.h;

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh,
                 "exhaustive motion search (%dx%d blocks, radius %d, %d seed "
                 "candidates)",
                 bs, bs, r, nseed);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, threads, 1, false, shmem.size)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create motion search shader, needs "
               "%d threads and %zu bytes of shared memory per work group!",
               threads, shmem.size);
        goto error;
    }

    PL_DEBUG(gpu, "pl_motion_estimate: searching %dx%d -> %dx%d blocks: block "
             "size %d, radius %d, %d candidates, %dx%d block tile, %d threads, "
             "%zu bytes of shared memory",
             tex_w, tex_h, grid_w, grid_h, bs, r, ncand, bpgx, bpgy, threads,
             shmem.size);

    // The rank of a candidate is its index in the LUT, derived from its thread
    // index, so the reduction can minimize the rank to resolve ties
    // deterministically.
    ident_t cand = sh_lut(sh, sh_lut_params(
        .object   = p->cand_lut,
        .var_type = PL_VAR_SINT,
        .lut_type = SH_LUT_UNIFORM,
        .method   = SH_LUT_NONE,
        .width    = ncand,
        .comps    = 2,
        .fill     = cand_lut_fill,
        .priv     = (void *) geom,
    ));

    if (!cand) {
        PL_ERR(gpu, "pl_motion_estimate: failed uploading candidate LUT!");
        goto error;
    }

    // Describe the textures for the reference frames and motion vector output.
    ident_t ref = bind_tex(sh, "ref", p->ref);
    ident_t tgt = bind_tex(sh, "tgt", p->tgt);
    ident_t out = bind_img(sh, "out_mv", p->out_mv);

    // Bind the flow field of the previous motion search image pyramid level,
    // unless this is the coarsest (first) level of the motion search.
    const bool has_parent = p->parent != NULL;
    const int parent_w = has_parent ? p->parent->params.w : 0;
    const int parent_h = has_parent ? p->parent->params.h : 0;
    const int pscale = p->parent_scale ? p->parent_scale : 2;
    const int rank_parent = geom->seeds.rank[MOTION_SEED_PARENT];
    const int rank_x = geom->seeds.rank[MOTION_SEED_PARENT_X];
    const int rank_y = geom->seeds.rank[MOTION_SEED_PARENT_Y];
    ident_t parent = NULL_IDENT;
    if (has_parent) {
        parent = bind_tex(sh, "parent", p->parent);
    }

    // Bind the flow field of the same motion search image pyramid level from
    // the previous motion search invocation on the previous frame pair. Don't
    // bind anything when there is no usable prior available (upon a cold
    // start, after a reset, or a change in motion estimation direction).
    const bool has_prior = p->prior != NULL;
    const int prior_w = has_prior ? p->prior->params.w : 0;
    const int prior_h = has_prior ? p->prior->params.h : 0;
    const int rank_prior = geom->seeds.rank[MOTION_SEED_PRIOR];
    ident_t prior = NULL_IDENT;
    if (has_prior) {
        prior = bind_tex(sh, "prior", p->prior);
    }

    // clang-format off
#pragma GLSLH /* motion_search_pass */                                          \
    /* Shared memory allocations and helper functions.                          \
     * TODO: see if we can convert to int8 luma beforehand, or if we can pack   \
     * multiple texels into a single load or shared memory bank. */             \
                                                                                \
    /* All pixels of the tile in the reference frame. */                        \
    shared uint sRef[${const int: ref_w * ref_h}];                              \
    /* Search windows in the target frame for each block of the tile. */        \
    shared uint sTgt[${const int: bpg * block_stride}];                         \
    /* Minimum cost of all candidates for each block. */                        \
    shared uint best_cost[${const int: bpg}];                                   \
    /* Deterministic tie breaking rank. */                                      \
    shared uint best_rank[${const int: bpg}];                                   \
    @if (subpixel) {                                                            \
        /* Cost one pixel before and after the winner of each block, on x and   \
         * then on y. */                                                        \
        shared uint near_cost[${const int: 4 * bpg}];                           \
    @}                                                                          \
    /* Sum of the cost of all candidates for each block. */                     \
    shared uint cost_sum[${const int: bpg}];                                    \
    /* Displacement into the target image that each block's search window will  \
     * be centered upon or seeded with. */                                      \
    shared ivec2 s_seed[${const int: bpg}];                                     \
    /* Avoid using any shared memory if there is only one seed candidate. */    \
    @if (nseed > 1) {                                                           \
        /* SSD of each seed candidate of each block of the tile. */             \
        shared uint s_score[${const int: bpg * nseed}];                         \
    @}                                                                          \
                                                                                \
    uint luma8b(float x) {                                                      \
        return uint(round(clamp(x, 0.0, 1.0) * 255.0));                         \
    }                                                                           \
                                                                                \
    @if (has_parent) {                                                          \
        /* Use the winning motion search candidate from the previous motion     \
         * search pyramid level. We assume that the resolution of the image     \
         * pyramid could be perfectly halved at each level, and that block      \
         * (x,y) in the parent level corresponds to blocks (i,j) in the current \
         * level for i in {2*x, 2*x+1}, j in {2*y, 2*y+1}. */                   \
        ivec2 parent_seed(ivec2 blk) {                                          \
            ivec2 p = clamp(blk / 2, ivec2(0),                                  \
                            ivec2(${const int: parent_w},                       \
                                  ${const int: parent_h}) - 1);                 \
            /* Multiply by two to account for change in scale between pyramid   \
             * levels. */                                                       \
            return ${const int: pscale} * ivec2(texelFetch($parent, p, 0).xy);  \
        }                                                                       \
    @}                                                                          \
                                                                                \
    @if (has_prior) {                                                           \
        /* Use the winning motion search candidate that was found by the        \
         * corresponding block in the same motion search pyramid level in the   \
         * invocation of the motion estimation on the previous frame pair. */   \
        ivec2 prior_seed(ivec2 blk) {                                           \
            ivec2 p = clamp(blk, ivec2(0),                                      \
                            ivec2(${const int: prior_w},                        \
                                  ${const int: prior_h}) - 1);                  \
            return ivec2(round(texelFetch($prior, p, 0).xy));                   \
        }                                                                       \
    @}                                                                          \
                                                                                \
    /* Sum of squared differences between reference block `b` and the block     \
     * at `offset` within its search window in shared memory. */                \
    uint window_ssd(int b, ivec2 offset) {                                      \
        int bx = b % ${const int: bpgx};                                        \
        int by = b / ${const int: bpgx};                                        \
        uint ssd = 0u;                                                          \
        for (int i = 0; i < ${const int: bs}; i++) {                            \
            int ref_row = (by * ${const int: bs} + i) * ${const int: ref_w}     \
                        + bx * ${const int: bs};                                \
            int tgt_row = b * ${const int: block_stride}                        \
                        + (offset.y + i) * ${const int: win_stride}             \
                        + offset.x;                                             \
            for (int j = 0; j < ${const int: bs}; j++) {                        \
                int diff = int(sRef[ref_row + j]) - int(sTgt[tgt_row + j]);     \
                ssd += uint(diff * diff);                                       \
            }                                                                   \
        }                                                                       \
        return ssd;                                                             \
    }                                                                           \
                                                                                \
    /* Returns the seed candidate that a thread `i` is responsible for          \
     * scoring. */                                                              \
    ivec2 seed_candidate(ivec2 blk, int i) {                                    \
        @if (has_parent) {                                                      \
            if (i == ${const int: rank_parent})                                 \
                return parent_seed(blk);                                        \
            /* The adjacent block towards the centre of the parent has the same \
             * parent, so only the one on the opposite side has a different one \
             * to seed with. */                                                 \
            ivec2 side = 2 * (blk & 1) - 1;                                     \
            if (i == ${const int: rank_x})                                      \
                return parent_seed(blk + ivec2(side.x, 0));                     \
            if (i == ${const int: rank_y})                                      \
                return parent_seed(blk + ivec2(0, side.y));                     \
        @}                                                                      \
        @if (has_prior) {                                                       \
            if (i == ${const int: rank_prior})                                  \
                return prior_seed(blk);                                         \
        @}                                                                      \
        return ivec2(0);                                                        \
    }                                                                           \
                                                                                \
    @if (nseed > 1 || subpixel) {                                               \
        /* Sum of squared differences between reference block `b` and the       \
         * corresponding block in the target frame, but displaced by `d`. */    \
        uint seed_score(int b, ivec2 blk, ivec2 d) {                            \
            ivec2 size = ivec2(${const int: tex_w}, ${const int: tex_h});       \
            int bx = b % ${const int: bpgx};                                    \
            int by = b / ${const int: bpgx};                                    \
            uint ssd = 0u;                                                      \
            for (int i = 0; i < ${const int: bs}; i++) {                        \
                int ref_row = (by * ${const int: bs} + i) * ${const int: ref_w} \
                            + bx * ${const int: bs};                            \
                for (int j = 0; j < ${const int: bs}; j++) {                    \
                    ivec2 pos = clamp(blk * ${const int: bs} + ivec2(j, i) + d, \
                                      ivec2(0), size - 1);                      \
                    int diff = int(sRef[ref_row + j])                           \
                             - int(luma8b(texelFetch($tgt, pos, 0).r));         \
                    ssd += uint(diff * diff);                                   \
                }                                                               \
            }                                                                   \
            return ssd;                                                         \
        }                                                                       \
    @}
    // clang-format on

    // clang-format off
#pragma GLSL /* motion_search_pass */                                           \
    uint tid = gl_LocalInvocationIndex;                                         \
    int b  = int(tid) / ${const int: tpb};  /* block within the tile */         \
    int c0 = int(tid) % ${const int: tpb};  /* first candidate of thread */     \
    /* The block of pixels within the window this thread will search. */        \
    int bx = b % ${const int: bpgx};                                            \
    int by = b / ${const int: bpgx};                                            \
    ivec2 tile = ivec2(gl_WorkGroupID.xy) * ivec2(${const int: bpgx},           \
                                                  ${const int: bpgy});          \
    /* The index of this thread's block of pixels within the grid of blocks. */ \
    ivec2 blk  = tile + ivec2(bx, by);                                          \
    ivec2 size = ivec2(${const int: tex_w}, ${const int: tex_h});               \
                                                                                \
    if (tid < ${const uint: bpg}) {                                             \
        best_cost[tid] = 0xFFFFFFFFu;                                           \
        best_rank[tid] = 0xFFFFFFFFu;                                           \
        cost_sum[tid]  = 0u;                                                    \
    }                                                                           \
                                                                                \
    /* Load block from reference image into shared memory. We clamp coordinates \
     * and pad partial blocks that extend past the image boundaries. */         \
    for (uint i = tid; i < ${const uint: ref_w * ref_h};                        \
         i += ${const uint: threads})                                           \
    {                                                                           \
        ivec2 elem = ivec2(int(i) % ${const int: ref_w},                        \
                           int(i) / ${const int: ref_w});                       \
        ivec2 ref_pos = clamp(tile * ${const int: bs} + elem, ivec2(0),         \
                              size - 1);                                        \
        sRef[i] = luma8b(texelFetch($ref, ref_pos, 0).r);                       \
    }                                                                           \
                                                                                \
    /* Select the displacement that block `b`s search window will be centered   \
     * on, saved to shared memory for the rest of the block to use. */          \
    ivec2 seed = ivec2(0);                                                      \
    @if (nseed > 1) {                                                           \
        barrier();  /* reference tile must be available in shared memory */     \
                                                                                \
        for (int i = c0; i < ${const int: nseed};                               \
             i += ${const int: tpb})                                            \
        {                                                                       \
            s_score[b * ${const int: nseed} + i] =                              \
                seed_score(b, blk, seed_candidate(blk, i));                     \
        }                                                                       \
        barrier();                                                              \
                                                                                \
        /* Pick the seed candidate with the lowest SSD. */                      \
        if (c0 == 0) {                                                          \
            int best_seed = 0;                                                  \
            uint best_ssd = s_score[b * ${const int: nseed}];                   \
            for (int i = 1; i < ${const int: nseed}; i++) {                     \
                uint seed_ssd = s_score[b * ${const int: nseed} + i];           \
                if (seed_ssd < best_ssd) {                                      \
                    best_seed = i;                                              \
                    best_ssd = seed_ssd;                                        \
                }                                                               \
            }                                                                   \
            seed = seed_candidate(blk, best_seed);                              \
            s_seed[b] = seed;                                                   \
        }                                                                       \
    @} else {                                                                   \
        if (c0 == 0) {                                                          \
            seed = seed_candidate(blk, 0);                                      \
            s_seed[b] = seed;                                                   \
        }                                                                       \
    @}                                                                          \
    barrier();                                                                  \
                                                                                \
    /* Load block from target image into shared memory. Each block can have a   \
     * unique search window based on `s_seed`, so account for that. Pad rows    \
     * of the search window in shared memory to avoid bank conflicts. */        \
    for (uint i = tid; i < ${const uint: bpg * win * win_stride};               \
         i += ${const uint: threads})                                           \
    {                                                                           \
        int win_idx = int(i) / ${const int: win * win_stride};                  \
        int j       = int(i) % ${const int: win * win_stride};                  \
        int win_x   = j % ${const int: win_stride};                             \
        int win_y   = j / ${const int: win_stride};                             \
        if (win_x < ${const int: win}) {                                        \
            ivec2 win_blk = tile + ivec2(win_idx % ${const int: bpgx},          \
                                         win_idx / ${const int: bpgx});         \
            ivec2 origin = win_blk * ${const int: bs} + s_seed[win_idx]         \
                         - ${const int: r};                                     \
            ivec2 tgt_pos = clamp(origin + ivec2(win_x, win_y), ivec2(0),       \
                                  size - 1);                                    \
            sTgt[win_idx * ${const int: block_stride}                           \
                 + win_y * ${const int: win_stride} + win_x]                    \
                = luma8b(texelFetch($tgt, tgt_pos, 0).r);                       \
        }                                                                       \
    }                                                                           \
    barrier();  /* make sure all elements are loaded into shared memory */      \
                                                                                \
    /* Compute the sum of squared differences (SSD) of all candidates of each   \
     * block. One thread loads all elements of a given candidate. */            \
    uint best = 0xFFFFFFFFu, rank = 0xFFFFFFFFu, sum = 0u;                      \
    for (int c = c0; c < ${const int: ncand}; c += ${const int: tpb}) {         \
        ivec2 cand_offset = $cand(c);                                           \
        /* Change the candidate offset into coordinates within the window */    \
        cand_offset += ${const int: r};                                         \
        uint ssd = window_ssd(b, cand_offset);                                  \
        /* For when a thread computes SSD of multiple candidates. */            \
        /* Strictly lower, so the order of candidates breaks ties. */           \
        if (ssd < best) {                                                       \
            best = ssd;                                                         \
            rank = uint(c);                                                     \
        }                                                                       \
        sum += ssd >> ${const int: cost_shift};                                 \
    }                                                                           \
                                                                                \
    /* Reduce over SSD cost. */                                                 \
    atomicMin(best_cost[b], best);                                              \
    atomicAdd(cost_sum[b], sum);                                                \
    barrier();                                                                  \
    /* Reduce over rank for deterministic minimum */                            \
    if (best == best_cost[b])                                                   \
        atomicMin(best_rank[b], rank);                                          \
    barrier();                                                                  \
    ivec2 winner = $cand(int(best_rank[b]));                                    \
    @if (subpixel) {                                                            \
        for (int i = c0; i < 4; i += ${const int: tpb}) {                       \
            ivec2 adj = winner;                                                 \
            adj[i / 2] += 2 * (i % 2) - 1;                                      \
            /* The search window does not extend beyond the candidates. */      \
            if (max(abs(adj.x), abs(adj.y)) <= ${const int: r}) {               \
                near_cost[4 * b + i] = window_ssd(b, adj + ${const int: r});    \
            } else {                                                            \
                near_cost[4 * b + i] = seed_score(b, blk, s_seed[b] + adj);     \
            }                                                                   \
        }                                                                       \
        barrier();                                                              \
    @}                                                                          \
    if (c0 == 0 && blk.x < ${const int: grid_w} &&                              \
        blk.y < ${const int: grid_h})                                           \
    {                                                                           \
        /* A single candidate has no SSD surface to judge. */                   \
        float q = 1.0;                                                          \
        @if (ncand > 1) {                                                       \
            float cost_min  = float(best_cost[b] >> ${const int: cost_shift});  \
            float cost_mean = float(cost_sum[b]) / float(${const int: ncand});  \
            q = 1.0 - (cost_min + ${const float: cost_noise})                   \
                    / (cost_mean + ${const float: cost_noise});                 \
            q = clamp(q, 0.0, 1.0);                                             \
        @}                                                                      \
        vec2 mv = vec2(seed + winner);                                          \
        @if (subpixel) {                                                        \
            /* Move to the vertex of the parabola through the cost of the       \
             * winner and of its two neighbours, separately for each axis. It   \
             * is within half a pixel if the winner is the lowest of the three, \
             * which the neighbours outside of the search need not respect. */  \
            vec2 prev = vec2(near_cost[4 * b], near_cost[4 * b + 2]);           \
            vec2 next = vec2(near_cost[4 * b + 1], near_cost[4 * b + 3]);       \
            vec2 curv = prev + next - vec2(2.0 * float(best_cost[b]));          \
            vec2 off = 0.5 * (prev - next) / max(curv, vec2(1.0));              \
            mv += clamp(off, vec2(-0.5), vec2(0.5));                            \
        @}                                                                      \
        imageStore($out, blk, vec4(mv, q, 0.0));                                \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = {
            PL_DIV_UP(grid_w, bpgx),
            PL_DIV_UP(grid_h, bpgy),
            1,
        },
    ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// EXPERIMENT: configurable smoothing, see /z/ai/claude/2026-11-12-residency.
// Env SMOOTH (all levels) / SMOOTH0 (level 0 override):
//   "<fallback>[,q=T][,novel=T]", fallback = median | fill | none.
// The own vector is kept if q >= T, or if its novelty
//   sum max(0, c - e_own) / (sum c + noise), c = min over the 8 neighbours,
// is >= T; otherwise the fallback is taken.
enum smooth_fb { SMOOTH_MEDIAN, SMOOTH_FILL, SMOOTH_NONE };
struct smooth_cfg { int fb; float qt, nt; int r; bool self; int sup; float coh, ot; };

static struct smooth_cfg smooth_cfg(bool level0)
{
    struct smooth_cfg c = { SMOOTH_FILL, -1, -1, 1, false, 0, 0, -1 };
    const char *s = level0 && getenv("SMOOTH0") ? getenv("SMOOTH0") : getenv("SMOOTH");
    if (!s)
        return c;
    if (!strncmp(s, "median", 6))
        c.fb = SMOOTH_MEDIAN;
    else if (!strncmp(s, "none", 4))
        c.fb = SMOOTH_NONE;
    const char *p;
    if ((p = strstr(s, "q=")))
        c.qt = atof(p + 2);
    if ((p = strstr(s, "novel=")))
        c.nt = atof(p + 6);
    if ((p = strstr(s, "r=")))
        c.r = atoi(p + 2);
    c.self = strstr(s, "self");
    if ((p = strstr(s, "sup=")))
        c.sup = atoi(p + 4);
    if ((p = strstr(s, "coh=")))
        c.coh = atof(p + 4);
    if ((p = strstr(s, "occ=")))
        c.ot = atof(p + 4);
    return c;
}

// Replace every block's motion vector with one derived from the surrounding 3x3
// neighbourhood of blocks, to suppress vectors that the search got wrong.
static bool motion_smooth_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                               pl_tex ref, pl_tex tgt, pl_tex in, pl_tex out,
                               bool level0, pl_tex raw_dbg)
{
    const int grid_w = out->params.w, grid_h = out->params.h;
    const int tex_w = ref->params.w, tex_h = ref->params.h;
    const struct smooth_cfg cfg = smooth_cfg(level0);
    const int fb = cfg.fb;
    const bool use_q = cfg.qt >= 0, use_nov = cfg.nt >= 0;
    const float qt = cfg.qt, nt = cfg.nt;
    const int r = cfg.r, nd = 2 * r + 1, nc = nd * nd, own_i = nc / 2;
    const bool self = cfg.self;
    const int skip_i = self ? -1 : own_i;
    const int sup = cfg.sup;
    // Penalty per px^2 of distance to the 3x3 median, capped at 4 px, in units
    // of a block SSD of one code value per pixel
    const float coh = cfg.coh * bs * bs / (255.0f * 255.0f);
    const bool use_coh = cfg.coh > 0;
    const bool use_occ = cfg.ot >= 0;
    const float ot = cfg.ot;
    const float noise = bs * bs * (0.5f / 255) * (0.5f / 255);

    // A work group covers `gx` x `gy` blocks, with one thread per row
    const int gx = PL_CLAMP(64 / bs, 1, 8), gy = 4;
    const int bw = gx * bs, bh = gy;
    const int ns = nc + 4;
    const size_t shmem = bw * bh * ns * sizeof(float);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector smoothing (%dx%d, %dx%d blocks, fb %d q %.2f nov %.2f self %d)",
                 nd, nd, grid_w, grid_h, fb, qt, nt, self);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, bw, bh, false, shmem)) {
        PL_ERR(gpu, "pl_motion_estimate: cannot create smoothing shader");
        goto error;
    }

    ident_t src = bind_tex(sh, "flow_in", in);
    ident_t dst = bind_img(sh, "flow_out", out);
    const bool has_dbg = raw_dbg;
    ident_t dbg = has_dbg ? bind_img(sh, "raw_dbg", raw_dbg) : NULL_IDENT;
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_pt, tgt_id = sh_bind(sh, tgt, PL_TEX_ADDRESS_CLAMP,
                                     PL_TEX_SAMPLE_LINEAR, "tgt", NULL, NULL,
                                     &tgt_pt);
    if (!tgt_id)
        goto error;

    // clang-format off
#pragma GLSLH /* motion_smooth_pass */                                          \
    shared float s_ssd[${const int: bw * bh * ns}];                             \
    void sort2(inout vec2 a, inout vec2 b) {                                    \
        vec2 lo = min(a, b);                                                    \
        b = max(a, b);                                                          \
        a = lo;                                                                 \
    }                                                                           \
    void sort9(inout vec2 a, inout vec2 b, inout vec2 c, inout vec2 d,          \
               inout vec2 e, inout vec2 f, inout vec2 g, inout vec2 h,          \
               inout vec2 i) {                                                  \
        sort2(c,f); sort2(a,d); sort2(e,i); sort2(b,h);                         \
        sort2(a,h); sort2(c,e); sort2(f,g); sort2(d,i);                         \
        sort2(a,c); sort2(b,d); sort2(e,f); sort2(h,i);                         \
        sort2(b,e); sort2(f,h); sort2(d,g);                                     \
        sort2(c,e); sort2(a,b); sort2(d,f); sort2(g,i);                         \
        sort2(c,d); sort2(e,f); sort2(g,h);                                     \
        sort2(b,c); sort2(d,e); sort2(f,g);                                     \
    }
    // clang-format on

    // clang-format off
#pragma GLSL /* motion_smooth_pass */                                           \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 size = ivec2(${const int: tex_w}, ${const int: tex_h});               \
    ivec2 tid  = ivec2(gl_LocalInvocationID.xy);                                \
    int row    = tid.x % ${const int: bs};                                      \
    ivec2 blk  = ivec2(gl_WorkGroupID.xy) *                                     \
                 ivec2(${const int: gx}, ${const int: gy}) +                    \
                 ivec2(tid.x / ${const int: bs}, tid.y);                        \
    int base   = (tid.y * ${const int: bw} + tid.x) * ${const int: ns};         \
                                                                                \
    vec2 mv[${const int: nc}];                                                  \
    for (int i = 0; i < ${const int: nc}; i++) {                                \
        ivec2 o = ivec2(i % ${const int: nd}, i / ${const int: nd}) - ${const int: r}; \
        ivec2 p = clamp(blk + o, ivec2(0), grid - 1);                           \
        mv[i] = texelFetch($src, p, 0).xy;                                      \
    }                                                                           \
                                                                                \
    float ssd[${const int: nc}];                                                \
    for (int k = 0; k < ${const int: nc}; k++)                                  \
        ssd[k] = 0.0;                                                           \
    float gain = 0.0, csum = 0.0, rs = 0.0, rs2 = 0.0;                          \
    for (int j = 0; j < ${const int: bs}; j++) {                                \
        ivec2 pos = min(blk * ${const int: bs} + ivec2(j, row), size - 1);      \
        float r = texelFetch($ref_id, pos, 0).r;                                \
        rs += r;                                                                \
        rs2 += r * r;                                                           \
        vec2 c = vec2(pos) + vec2(0.5);                                         \
        float cmin = 1e30, eown = 0.0;                                          \
        for (int k = 0; k < ${const int: nc}; k++) {                            \
            float d = r - textureLod($tgt_id, (c + mv[k]) * $tgt_pt, 0.0).r;    \
            ssd[k] += d * d;                                                    \
            ivec2 o = ivec2(k % ${const int: nd}, k / ${const int: nd}) - ${const int: r}; \
            if (k == ${const int: own_i})                                       \
                eown = d * d;                                                   \
            else if (all(lessThanEqual(abs(o), ivec2(1))))                      \
                cmin = min(cmin, d * d);                                        \
        }                                                                       \
        gain += max(0.0, cmin - eown);                                          \
        csum += cmin;                                                           \
    }                                                                           \
    for (int k = 0; k < ${const int: nc}; k++)                                  \
        s_ssd[base + k] = ssd[k];                                               \
    s_ssd[base + ${const int: nc}] = gain;                                      \
    s_ssd[base + ${const int: nc + 1}] = csum;                                  \
    s_ssd[base + ${const int: nc + 2}] = rs;                                    \
    s_ssd[base + ${const int: nc + 3}] = rs2;                                   \
    barrier();                                                                  \
                                                                                \
    if (all(lessThan(blk, grid)) && row == 0) {                                 \
        vec4 own = texelFetch($src, blk, 0);                                    \
        @if (has_dbg)                                                           \
            imageStore($dbg, blk, own);                                         \
        vec2 res = own.xy;                                                      \
        vec2 m[9];                                                              \
        for (int i = 0; i < 9; i++)                                             \
            m[i] = texelFetch($src, clamp(blk + ivec2(i % 3, i / 3) - 1,        \
                                          ivec2(0), grid - 1), 0).xy;           \
        sort9(m[0], m[1], m[2], m[3], m[4], m[5], m[6], m[7], m[8]);            \
        @if (fb == SMOOTH_MEDIAN)                                               \
            res = m[4];                                                         \
        @if (fb == SMOOTH_FILL) {                                               \
            float best_ssd = 1e30;                                              \
            for (int i = 0; i < ${const int: nc}; i++) {                        \
                ivec2 p = blk + ivec2(i % ${const int: nd}, i / ${const int: nd}) - ${const int: r}; \
                float sum = 0.0;                                                \
                for (int j = 0; j < ${const int: bs}; j++)                      \
                    sum += s_ssd[base + j * ${const int: ns} + i];              \
                bool inside = all(equal(p, clamp(p, ivec2(0), grid - 1)));      \
                bool skip = i == ${const int: skip_i};                          \
                @if (use_coh) {                                                 \
                    vec2 dm = mv[i] - m[4];                                     \
                    sum += ${const float: coh} * min(dot(dm, dm), 16.0);        \
                @}                                                              \
                if (inside && !skip && sum < best_ssd) {                        \
                    res = mv[i];                                                \
                    best_ssd = sum;                                             \
                }                                                               \
            }                                                                   \
            @if (use_occ) {                                                     \
                float t1 = 0.0, t2 = 0.0;                                       \
                for (int j = 0; j < ${const int: bs}; j++) {                    \
                    t1 += s_ssd[base + j * ${const int: ns} + ${const int: nc + 2}]; \
                    t2 += s_ssd[base + j * ${const int: ns} + ${const int: nc + 3}]; \
                }                                                               \
                float energy = t2 - t1 * t1 / ${const float: bs * bs};          \
                if (best_ssd > ${const float: ot} * (energy + ${const float: noise})) \
                    res = m[4];                                                 \
            @}                                                                  \
        @}                                                                      \
        bool keep = false;                                                      \
        @if (use_q)                                                             \
            keep = keep || own.z >= ${const float: qt};                         \
        @if (use_nov) {                                                         \
            float g = 0.0, cs = 0.0;                                            \
            for (int j = 0; j < ${const int: bs}; j++) {                        \
                g  += s_ssd[base + j * ${const int: ns} + ${const int: nc}];    \
                cs += s_ssd[base + j * ${const int: ns} + ${const int: nc + 1}]; \
            }                                                                   \
            keep = keep || g / (cs + ${const float: noise}) >= ${const float: nt}; \
        @}                                                                      \
        @if (sup > 0) {                                                         \
            int n_sup = 0;                                                      \
            for (int i = 0; i < 9; i++) {                                       \
                ivec2 p = blk + ivec2(i % 3, i / 3) - 1;                        \
                bool inside = all(equal(p, clamp(p, ivec2(0), grid - 1)));      \
                vec2 dv = texelFetch($src, clamp(p, ivec2(0), grid - 1), 0).xy - own.xy; \
                if (i != 4 && inside && dot(dv, dv) <= 1.0)                     \
                    n_sup++;                                                    \
            }                                                                   \
            keep = keep || n_sup >= ${const int: sup};                          \
        @}                                                                      \
        if (keep)                                                               \
            res = own.xy;                                                       \
        imageStore($dst, blk, vec4(res, 1.0, 0.0));                             \
    }
    // clang-format on

    const struct pl_dispatch_compute_params *params = pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = {
            PL_DIV_UP(grid_w, gx),
            PL_DIV_UP(grid_h, gy),
            1,
        },
    );

    return pl_dispatch_compute(dispatch, params);

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Writes a vector for every pixel of `out`, which measures the full
// resolution: the vector of the pixel's block in `flow`, or that of one of the
// 8 neighbouring blocks if it matches the window of pixels around the pixel
// better by more than `MOTION_REFINE_MARGIN`. This follows motion boundaries
// within a block, which a single vector per block cannot.
//
// The confidence estimates whether the pixel is visible in the other frame. It
// falls with the mismatch: the geometric mean of the SSD of the vector taken
// and the mean SSD of all candidates, relative to the texture energy of the
// window. Including the other candidates makes this larger at motion
// boundaries, where occlusions happen. Vectors that lead out of the frame get
// no confidence at all.
static bool motion_refine_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                               pl_tex ref, pl_tex tgt, pl_tex flow, pl_tex out,
                               bool plain)
{
    const int w = out->params.w, h = out->params.h;
    const int grid_w = flow->params.w, grid_h = flow->params.h;
    const int win = MOTION_REFINE_WINDOW;
    // Work groups cover at most one block, so they share the candidates
    const int dim = PL_MIN(bs, 16), n = PL_DIV_UP(bs, dim);
    // Squared differences over the windows of all pixels of the work group
    const int tile = dim + win - 1;
    const int per_thread = PL_DIV_UP(PL_SQUARE(tile), PL_SQUARE(dim));

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector refinement (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;

    const size_t shmem = PL_SQUARE(tile) * sizeof(float);
    if (!sh_try_compute(sh, dim, dim, false, shmem)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create motion vector refinement "
               "shader, needs %dx%d threads per work group!", dim, dim);
        goto error;
    }

    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t dst = bind_img(sh, "flow_out", out);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    // The vectors are fractional
    ident_t tgt_pt, tgt_id = sh_bind(sh, tgt, PL_TEX_ADDRESS_CLAMP,
                                     PL_TEX_SAMPLE_LINEAR, "tgt", NULL, NULL,
                                     &tgt_pt);
    if (!tgt_id)
        goto error;

    // SSD over the window, in units of the luma textures
    const float scale = PL_SQUARE(win) / (255.0f * 255.0f);
    const float margin = b_envf("LM_B_MARGIN", MOTION_REFINE_MARGIN) * scale;
    const float flat = MOTION_REFINE_FLAT * scale;

    // clang-format off
#pragma GLSLH /* motion_refine_pass */                                          \
    shared float s_diff[${const int: tile * tile}];
#pragma GLSL /* motion_refine_pass */                                           \
    ivec2 size  = ivec2(${const int: w}, ${const int: h});                      \
    ivec2 grid  = ivec2(${const int: grid_w}, ${const int: grid_h});            \
    ivec2 local = ivec2(gl_LocalInvocationID.xy);                               \
    ivec2 blk   = ivec2(gl_WorkGroupID.xy) / ${const int: n};                   \
    ivec2 sub   = ivec2(gl_WorkGroupID.xy) % ${const int: n} * ${const int: dim}\
                + local;                                                        \
    ivec2 pos   = blk * ${const int: bs} + sub;                                 \
                                                                                \
    float r[${const int: per_thread}];                                          \
    vec2 c[${const int: per_thread}];                                           \
    for (int l = 0; l < ${const int: per_thread}; l++) {                        \
        int j = int(gl_LocalInvocationIndex) + l * ${const int: dim * dim};     \
        ivec2 q = clamp(pos - local + ivec2(j % ${const int: tile},             \
                                            j / ${const int: tile})             \
                        - ${const int: win / 2}, ivec2(0), size - 1);           \
        r[l] = texelFetch($ref_id, q, 0).r;                                     \
        c[l] = vec2(q) + vec2(0.5);                                             \
    }                                                                           \
                                                                                \
    /* The block's own vector first, then those of its neighbours. */           \
    vec3 cand[9];                                                               \
    for (int k = 0; k < 9; k++) {                                               \
        int i = k == 0 ? 4 : k <= 4 ? k - 1 : k;                                \
        ivec2 p = clamp(blk + ivec2(i % 3, i / 3) - 1, ivec2(0), grid - 1);     \
        cand[k] = texelFetch($src, p, 0).xyz;                                   \
    }                                                                           \
                                                                                \
    float energy = 0.0;                                                         \
    @if (!plain) {                                                              \
    /* Texture energy of the window: its squared differences to its centre */   \
    for (int l = 0; l < ${const int: per_thread}; l++) {                        \
        int j = int(gl_LocalInvocationIndex) + l * ${const int: dim * dim};     \
        if (j < ${const int: tile * tile})                                      \
            s_diff[j] = r[l];                                                   \
    }                                                                           \
    barrier();                                                                  \
    float centre = s_diff[(local.y + ${const int: win / 2}) * ${const int: tile}\
                          + local.x + ${const int: win / 2}];                   \
    for (int j = 0; j < ${const int: win * win}; j++) {                         \
        float d = s_diff[(local.y + j / ${const int: win}) * ${const int: tile} \
                         + local.x + j % ${const int: win}] - centre;           \
        energy += d * d;                                                        \
    }                                                                           \
    @}                                                                          \
                                                                                \
    vec3 best = cand[0];                                                        \
    float best_ssd = 0.0, best_raw = 0.0, mean_ssd = 0.0;                       \
    for (int k = 0; k < 9; k++) {                                               \
        barrier();                                                              \
        for (int l = 0; l < ${const int: per_thread}; l++) {                    \
            int j = int(gl_LocalInvocationIndex) + l * ${const int: dim * dim}; \
            vec2 pt = (c[l] + cand[k].xy) * $tgt_pt;                            \
            float d = r[l] - textureLod($tgt_id, pt, 0.0).r;                    \
            if (j < ${const int: tile * tile})                                  \
                s_diff[j] = d * d;                                              \
        }                                                                       \
        barrier();                                                              \
                                                                                \
        float raw = 0.0;                                                        \
        for (int j = 0; j < ${const int: win * win}; j++) {                     \
            raw += s_diff[(local.y + j / ${const int: win}) * ${const int: tile}\
                          + local.x + j % ${const int: win}];                   \
        }                                                                       \
        mean_ssd += raw / 9.0;                                                  \
        float ssd = k == 0 ? raw : raw + ${float: margin};                      \
        if (k == 0 || ssd < best_ssd) {                                         \
            best = cand[k];                                                     \
            best_ssd = ssd;                                                     \
            best_raw = raw;                                                     \
        }                                                                       \
    }                                                                           \
                                                                                \
    float mismatch = sqrt(best_raw * mean_ssd) / (energy + ${float: flat});     \
    float conf = 1.0 / (1.0 + mismatch / ${float: MOTION_REFINE_OCCLUSION});    \
    vec2 to = vec2(pos) + vec2(0.5) + best.xy;                                  \
    if (any(lessThan(to, vec2(0.0))) || any(greaterThan(to, vec2(size))))      \
        conf = 0.0;                                                             \
                                                                                \
    if (all(lessThan(sub, ivec2(${const int: bs}))) && all(lessThan(pos, size)))\
        imageStore($dst, pos, vec4(best.xy, conf, 0.0));
    // clang-format on

    const struct pl_dispatch_compute_params *params = pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = { grid_w * n, grid_h * n, 1 },
    );

    return pl_dispatch_compute(dispatch, params);

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Set by `pl_shader_frame_interp`: whether the estimation runs stage B's
// occlusion and fill passes after the per pixel selection.
static int motion_b_request;

// EXPERIMENT (stage B): per pixel selection among the vectors of the pixel's
// block, its 8 neighbours, blocks further away along the axes and the zero
// vector, by the difference over a window in which each pixel counts by its
// similarity to the centre (adaptive support weights).
static bool motion_select_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                               pl_tex ref, pl_tex tgt, pl_tex blocks, pl_tex out,
                               pl_tex prev_field)
{
    const int w = out->params.w, h = out->params.h;
    const int win = dc_env("LM_B_SW", 5);
    const float sigma = b_envf("LM_B_SSIG", 100.0f);
    const int far = dc_env("LM_B_SFAR", 4);   // far rings: 2, 4, 8 blocks
    const bool zero = dc_env("LM_B_SZERO", 1);
    const float margin = b_envf("LM_B_SMARGIN", 1.0f) / 255.0f;
    const int nring = dc_env("LM_B_SDIAG", 1) ? 8 : 4;
    const bool temp = prev_field && dc_env("LM_B_STEMP", 1);
    const int ncand = 9 + nring * far + (zero ? 1 : 0) + (temp ? 1 : 0);
    const int l2 = dc_env("LM_B_SL2", 1);
    const bool asw = sigma < 10.0f;
    const float dup = PL_SQUARE(b_envf("LM_B_SDUP", 0.1f));

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector selection (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;
    const int dim = PL_MIN(bs, 16), sub = PL_DIV_UP(bs, dim);
    if (!sh_try_compute(sh, dim, dim, false, 0))
        goto error;

    ident_t blk = bind_tex(sh, "blocks", blocks);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex_linear(sh, "tgt", tgt);
    ident_t dst = bind_img(sh, "flow_out", out);
    ident_t prev_id = temp ? bind_tex(sh, "prev_field", prev_field) : NULL_IDENT;

    // clang-format off
#pragma GLSL /* motion_select_pass */                                           \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 grid = ivec2(${const int: blocks->params.w},                          \
                       ${const int: blocks->params.h});                         \
    /* one block per work group (or a part of it), which shares candidates */   \
    ivec2 wg   = ivec2(gl_WorkGroupID.xy);                                      \
    ivec2 lpos = wg % ${const int: sub} * ${const int: dim} + ivec2(gl_LocalInvocationID.xy);\
    ivec2 pos  = wg / ${const int: sub} * ${const int: bs} + lpos;              \
    if (all(lessThan(pos, size)) && all(lessThan(lpos, ivec2(${const int: bs})))) {\
        const int W = ${const int: win};                                        \
        vec2 pt = 1.0 / vec2(size);                                             \
        float centre = texelFetch($ref_id, pos, 0).r;                           \
        float wsum = 0.0;                                                       \
        @if (asw) {                                                             \
        for (int i = 0; i < W * W; i++) {                                       \
            ivec2 q = clamp(pos + ivec2(i % W, i / W) - W / 2, ivec2(0), size - 1);\
            wsum += exp(-abs(texelFetch($ref_id, q, 0).r - centre) / ${const float: sigma});\
        }                                                                       \
        @} else {                                                               \
        wsum = float(W * W);                                                    \
        @}                                                                      \
        const ivec2 near[9] = ivec2[9](ivec2(0, 0),                             \
            ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0),            \
            ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));               \
        vec2 best = vec2(0.0), own = vec2(0.0), last = vec2(0.0);               \
        float best_cost = 1e9;                                                  \
        for (int k = 0; k < ${const int: ncand}; k++) {                         \
            vec2 v = vec2(0.0);                                                 \
            if (k < ${const int: 9 + nring * far}) {                            \
                ivec2 o = near[k < 9 ? k : 0];                                  \
                if (k >= 9) {                                                   \
                    int j = (k - 9) % ${const int: nring};                      \
                    @if (nring == 4)                                            \
                        j = j == 0 ? 2 : j == 1 ? 4 : j == 2 ? 5 : 7;           \
                    @else                                                       \
                        j += 1;                                                 \
                    o = near[j] * (2 << ((k - 9) / ${const int: nring}));       \
                }                                                               \
                ivec2 b = clamp(pos / ${const int: bs} + o, ivec2(0), grid - 1);\
                v = texelFetch($blk, b, 0).xy;                                  \
            }                                                                   \
            @if (temp) {                                                        \
            if (k == ${const int: ncand - 1})                                   \
                v = texelFetch($prev_id, pos, 0).xy;                            \
            @}                                                                  \
            /* most candidates repeat the block's own or the previous one */    \
            vec2 d0 = v - own, d1 = v - last, d2 = v - best;                    \
            if (k > 0 && (dot(d0, d0) < ${const float: dup} || dot(d1, d1) < ${const float: dup} || dot(d2, d2) < ${const float: dup}))\
                continue;                                                       \
            if (k == 0)                                                         \
                own = v;                                                        \
            last = v;                                                           \
            float cost = 0.0;                                                   \
            for (int i = 0; i < W * W; i++) {                                   \
                ivec2 q = clamp(pos + ivec2(i % W, i / W) - W / 2, ivec2(0), size - 1);\
                float ri = texelFetch($ref_id, q, 0).r;                         \
                float d = ri - textureLod($tgt_id, (vec2(q) + 0.5 + v) * pt, 0.0).r;\
                float wi = 1.0;                                                 \
                @if (asw)                                                       \
                    wi = exp(-abs(ri - centre) / ${const float: sigma});        \
                cost += wi * (${const int: l2} != 0 ? d * d * 255.0 : abs(d));  \
            }                                                                   \
            cost = cost / wsum + (k == 0 ? 0.0 : ${const float: margin});       \
            if (cost < best_cost) {                                             \
                best_cost = cost;                                               \
                best = v;                                                       \
            }                                                                   \
        }                                                                       \
        imageStore($dst, pos, vec4(best, 1.0, 0.0));                            \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { blocks->params.w * sub, blocks->params.h * sub, 1 }));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// EXPERIMENT (stage B): the selection window drags the pixels next to a
// motion boundary along with the side of higher contrast. Where a pixel's
// neighbours move differently, let it take a neighbour's vector if the pixel
// itself matches clearly better that way.
// EXPERIMENT (two step B): per S x S sub-block, the best of the candidates of
// `motion_select_pass` (block vectors of `blocks` around it, far rings, zero,
// previous field) by the SSD over the sub-block and a border. Writes one
// vector per sub-block into `out`.
static bool motion_subsel_pass(pl_gpu gpu, pl_dispatch dispatch, int bs, int S,
                               pl_tex ref, pl_tex tgt, pl_tex blocks, pl_tex out,
                               pl_tex prev_field)
{
    const int w = ref->params.w, h = ref->params.h;
    const int gw = out->params.w, gh = out->params.h;
    const int far = dc_env("LM_B2_FAR", 4);
    const bool zero = dc_env("LM_B2_ZERO", 1);
    const float margin = b_envf("LM_B2_MARGIN", 1.0f) / 255.0f;
    const int nring = dc_env("LM_B2_DIAG", 1) ? 8 : 4;
    const bool temp = prev_field && dc_env("LM_B2_TEMP", 1);
    const int border = dc_env("LM_B2_BORDER", 1);
    const int ncand = 9 + nring * far + (zero ? 1 : 0) + (temp ? 1 : 0);
    const float dup = PL_SQUARE(b_envf("LM_B2_DUP", 0.1f));
    const int W = S + 2 * border;

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion sub-block selection (%dx%d, %dx%d)", gw, gh, S, S);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 8, 8, false, 0)) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }
    ident_t blk = bind_tex(sh, "blocks", blocks);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex_linear(sh, "tgt", tgt);
    ident_t dst = bind_img(sh, "sub_out", out);
    ident_t prev_id = temp ? bind_tex(sh, "prev_field", prev_field) : NULL_IDENT;
#pragma GLSL /* motion_subsel_pass */                                           \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 grid = ivec2(${const int: blocks->params.w}, ${const int: blocks->params.h});\
    ivec2 sb   = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(sb, ivec2(${const int: gw}, ${const int: gh})))) {         \
        ivec2 org = sb * ${const int: S};                                       \
        ivec2 cen = min(org + ${const int: S} / 2, size - 1);                   \
        vec2 pt = 1.0 / vec2(size);                                             \
        const ivec2 near[9] = ivec2[9](ivec2(0, 0),                             \
            ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0),            \
            ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));               \
        vec2 best = vec2(0.0), own = vec2(0.0), last = vec2(0.0);               \
        float best_cost = 1e9;                                                  \
        for (int k = 0; k < ${const int: ncand}; k++) {                         \
            vec2 v = vec2(0.0);                                                 \
            if (k < ${const int: 9 + nring * far}) {                            \
                ivec2 o = near[k < 9 ? k : 0];                                  \
                if (k >= 9) {                                                   \
                    int j = (k - 9) % ${const int: nring};                      \
                    @if (nring == 4)                                            \
                        j = j == 0 ? 2 : j == 1 ? 4 : j == 2 ? 5 : 7;           \
                    @else                                                       \
                        j += 1;                                                 \
                    o = near[j] * (2 << ((k - 9) / ${const int: nring}));       \
                }                                                               \
                ivec2 b = clamp(cen / ${const int: bs} + o, ivec2(0), grid - 1);\
                v = texelFetch($blk, b, 0).xy;                                  \
            }                                                                   \
            @if (temp) {                                                        \
            if (k == ${const int: ncand - 1})                                   \
                v = texelFetch($prev_id, cen, 0).xy;                            \
            @}                                                                  \
            vec2 d0 = v - own, d1 = v - last, d2 = v - best;                    \
            if (k > 0 && (dot(d0, d0) < ${const float: dup} || dot(d1, d1) < ${const float: dup} || dot(d2, d2) < ${const float: dup}))\
                continue;                                                       \
            if (k == 0)                                                         \
                own = v;                                                        \
            last = v;                                                           \
            float cost = 0.0;                                                   \
            for (int i = 0; i < ${const int: W * W}; i++) {                     \
                ivec2 q = clamp(org + ivec2(i % ${const int: W}, i / ${const int: W}) - ${const int: border}, ivec2(0), size - 1);\
                float d = texelFetch($ref_id, q, 0).r - textureLod($tgt_id, (vec2(q) + 0.5 + v) * pt, 0.0).r;\
                cost += d * d * 255.0;                                          \
            }                                                                   \
            cost = cost / float(${const int: W * W}) + (k == 0 ? 0.0 : ${const float: margin});\
            if (cost < best_cost) {                                             \
                best_cost = cost;                                               \
                best = v;                                                       \
            }                                                                   \
        }                                                                       \
        imageStore($dst, sb, vec4(best, 1.0, 0.0));                             \
    }
    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh, .dispatch_size = { PL_DIV_UP(gw, 8), PL_DIV_UP(gh, 8), 1 }));
}

// EXPERIMENT: nearest upsampling of a sub-block field to full resolution
static bool motion_upsample_pass(pl_gpu gpu, pl_dispatch dispatch, int S,
                                 pl_tex sub, pl_tex out)
{
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describe(sh, "motion sub-block upsampling");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }
    ident_t src = bind_tex(sh, "sub", sub);
    ident_t dst = bind_img(sh, "up_out", out);
#pragma GLSL /* upsample */                                                     \
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);                                \
    if (all(lessThan(pos, ivec2(${const int: out->params.w}, ${const int: out->params.h}))))\
        imageStore($dst, pos, vec4(texelFetch($src, pos / ${const int: S}, 0).xy, 1.0, 0.0));
    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh, .dispatch_size = { PL_DIV_UP(out->params.w, 16), PL_DIV_UP(out->params.h, 16), 1 }));
}

static bool motion_sharpen_pass(pl_gpu gpu, pl_dispatch dispatch, pl_tex ref,
                                pl_tex tgt, pl_tex flow, pl_tex out)
{
    const int w = out->params.w, h = out->params.h;
    const int rad = dc_env("LM_B_SHR", 2);
    const float tau = b_envf("LM_B_SHT", 4.0f) / 255.0f;
    const float rel = b_envf("LM_B_SHREL", 0.5f);
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion boundary sharpening (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0))
        goto error;

    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex_linear(sh, "tgt", tgt);
    ident_t dst = bind_img(sh, "flow_out", out);

    // clang-format off
#pragma GLSL /* motion_sharpen_pass */                                          \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(pos, size))) {                                             \
        vec2 pt = 1.0 / vec2(size);                                             \
        vec4 f = texelFetch($src, pos, 0);                                      \
        float centre = texelFetch($ref_id, pos, 0).r;                           \
        float own = -1.0, best = 0.0;                                           \
        vec2 pick = f.xy;                                                       \
        vec2 last = f.xy;                                                       \
        for (int r = 1; r <= ${const int: rad}; r++) {                          \
            for (int i = 0; i < 9; i++) {                                       \
                if (i == 4)                                                     \
                    continue;                                                   \
                ivec2 q = clamp(pos + (ivec2(i % 3, i / 3) - 1) * r, ivec2(0), size - 1);\
                vec2 v = texelFetch($src, q, 0).xy;                             \
                vec2 d = v - f.xy, e = v - last;                                \
                if (dot(d, d) <= 0.25 || dot(e, e) <= 0.01)                     \
                    continue;                                                   \
                last = v;                                                       \
                if (own < 0.0) {                                                \
                    own = abs(centre - textureLod($tgt_id, (vec2(pos) + 0.5 + f.xy) * pt, 0.0).r);\
                    best = own * ${const float: rel} - ${const float: tau};     \
                }                                                               \
                float c = abs(centre - textureLod($tgt_id, (vec2(pos) + 0.5 + v) * pt, 0.0).r);\
                if (c < best) {                                                 \
                    best = c;                                                   \
                    pick = v;                                                   \
                }                                                               \
            }                                                                   \
        }                                                                       \
        imageStore($dst, pos, vec4(pick, f.zw));                                \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Stage B: marks the pixels of `flow` that are hidden in the other frame, in
// the z channel of `out` (visibility, hidden below 0.5). A pixel is hidden if
// following its vector and then the opposite field's (`opp`) does not lead
// back to it, and the content there differs from its own in `ref`.
static bool motion_occlusion_pass(pl_gpu gpu, pl_dispatch dispatch, pl_tex ref,
                                  pl_tex flow, pl_tex opp, pl_tex out)
{
    const int w = out->params.w, h = out->params.h;
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion occlusion (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0))
        goto error;

    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t opp_id = bind_tex_linear(sh, "flow_opp", opp);
    const int omode = dc_env("LM_B_OMODE", 1);
    ident_t ref_id = bind_tex_linear(sh, "ref", ref);
    ident_t dst = bind_img(sh, "flow_out", out);
    const float fbt = b_envf("LM_B_FBT", 1.5f), ot = b_envf("LM_B_OT", 0.04f);
    const float noise = 9.0f / (255.0f * 255.0f);

    // clang-format off
#pragma GLSL /* motion_occlusion_pass */                                        \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(pos, size))) {                                             \
        vec2 pt = 1.0 / vec2(size);                                             \
        vec2 f = texelFetch($src, pos, 0).xy;                                   \
        vec2 rt = f + textureLod($opp_id, (vec2(pos) + 0.5 + f) * pt, 0.0).xy;  \
        @if (omode >= 1) {                                                      \
        /* any of the pixels around the landing point may be the counterpart */ \
        ivec2 base = ivec2(floor(vec2(pos) + f)) - ${const int: omode - 1};     \
        for (int i = 0; i < ${const int: (omode + 1) * (omode + 1)}; i++) {     \
            ivec2 q = clamp(base + ivec2(i % ${const int: omode + 1}, i / ${const int: omode + 1}), ivec2(0), size - 1);\
            vec2 r2 = f + texelFetch($opp_id, q, 0).xy;                         \
            if (dot(r2, r2) < dot(rt, rt))                                      \
                rt = r2;                                                        \
        }                                                                       \
        @}                                                                      \
        float vis = 1.0 / (1.0 + dot(rt, rt) / ${const float: fbt * fbt});      \
        if (vis < 0.5) {                                                        \
            float centre = texelFetch($ref_id, pos, 0).r;                       \
            float ssd = 0.0, energy = 0.0;                                      \
            for (int j = 0; j < 9; j++) {                                       \
                ivec2 q = clamp(pos + ivec2(j % 3, j / 3) - 1, ivec2(0), size - 1);\
                float r = texelFetch($ref_id, q, 0).r;                          \
                float s = textureLod($ref_id, (vec2(q) + 0.5 + rt) * pt, 0.0).r;\
                ssd += (r - s) * (r - s);                                       \
                energy += (r - centre) * (r - centre);                          \
            }                                                                   \
            float m = ssd / (energy + ${const float: noise}) / ${const float: ot};\
            vis = max(vis, 1.0 / (1.0 + m * m));                                \
        }                                                                       \
        imageStore($dst, pos, vec4(f, vis, 0.0));                               \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Matching cannot find the vector of a pixel hidden in the other frame. Give
// it the vector of the nearest visible pixels around it instead: those found
// in each of 8 directions vote, weighted by their similarity in brightness and
// their distance, for the vector most of them (within 1 px) move with.
//
// Also stores how well each pixel matches the one its vector leads to in
// `tgt`, as the absolute difference, in the w channel.
static bool motion_fill_pass(pl_gpu gpu, pl_dispatch dispatch, pl_tex ref,
                              pl_tex tgt, pl_tex flow, pl_tex out)
{
    const int w = out->params.w, h = out->params.h;
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector fill (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0))
        goto error;

    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex_linear(sh, "tgt", tgt);
    ident_t dst = bind_img(sh, "flow_out", out);
    const int wmode = dc_env("LM_B_WMODE", 2);
    const float wnorm = b_envf("LM_B_WNORM", 1.0f);

    // clang-format off
#pragma GLSL /* motion_fill_pass */                                             \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(pos, size))) {                                             \
        vec4 f = texelFetch($src, pos, 0);                                      \
        float centre = texelFetch($ref_id, pos, 0).r;                           \
        if (f.z < 0.5) {                                                        \
            /* Distances along the axes, and along the diagonals */             \
            const int ax[9] = int[9](1, 2, 3, 5, 8, 12, 16, 24, 32);            \
            const int dg[9] = int[9](1, 1, 2, 4, 6, 8, 11, 17, 23);             \
            const ivec2 dirs[8] = ivec2[8](ivec2(1, 0), ivec2(1, 1),            \
                ivec2(0, 1), ivec2(-1, 1), ivec2(-1, 0), ivec2(-1, -1),         \
                ivec2(0, -1), ivec2(1, -1));                                    \
            vec2 cand[8];                                                       \
            float weight[8];                                                    \
            int num = 0;                                                        \
            for (int a = 0; a < 8; a++) {                                       \
                for (int r = 0; r < 9; r++) {                                   \
                    ivec2 o = dirs[a] * ((a & 1) == 1 ? dg[r] : ax[r]);         \
                    ivec2 s = clamp(pos + o, ivec2(0), size - 1);               \
                    vec4 n = texelFetch($src, s, 0);                            \
                    if (n.z < 0.5)                                              \
                        continue;                                               \
                    float d = (texelFetch($ref_id, s, 0).r - centre)            \
                            / ${const float: MOTION_FILL_SIGMA};                \
                    weight[num] = exp(-d * d) / (1.0 + length(vec2(o)) / 4.0);  \
                    cand[num++] = n.xy;                                         \
                    break;                                                      \
                }                                                               \
            }                                                                   \
                                                                                \
            float best = 0.0;                                                   \
            for (int k = 0; k < num; k++) {                                     \
                float score = 0.0;                                              \
                for (int j = 0; j < num; j++) {                                 \
                    vec2 d = cand[k] - cand[j];                                 \
                    if (dot(d, d) <= 1.0)                                       \
                        score += weight[j];                                     \
                }                                                               \
                if (score > best) {                                             \
                    best = score;                                               \
                    f.xy = cand[k];                                             \
                }                                                               \
            }                                                                   \
        }                                                                       \
                                                                                \
        vec2 to = (vec2(pos) + vec2(0.5) + f.xy) / vec2(size);                  \
        f.w = abs(centre - textureLod($tgt_id, to, 0.0).r);                     \
        @if (wmode > 0) {                                                       \
        float ssd = 0.0, energy = 0.0;                                          \
        for (int j = 0; j < 9; j++) {                                           \
            ivec2 q = clamp(pos + ivec2(j % 3, j / 3) - 1, ivec2(0), size - 1); \
            float r = texelFetch($ref_id, q, 0).r;                              \
            float d = r - textureLod($tgt_id, (vec2(q) + vec2(0.5) + f.xy) / vec2(size), 0.0).r;\
            ssd += d * d;                                                       \
            energy += (r - centre) * (r - centre);                              \
        }                                                                       \
        f.w = ${const int: wmode} == 1 ? sqrt(ssd / 9.0)                        \
            : 0.25 * min(sqrt(ssd / (energy + ${const float: 9.0f / (255.0f * 255.0f)})) / ${const float: wnorm}, 1.0);\
        @}                                                                      \
        imageStore($dst, pos, f);                                               \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// The splat moves every pixel of both frames along its vector to where it is
// at the time of the intermediate frame, and keeps one of the pixels arriving
// at each position: the smallest key, by atomicMin(). Bits of a key:
//   31-30  3 - layer (1 for pixels known to pass in front of others, else 0)
//   29     hidden in the other frame
//   28-21  mismatch with the other frame
//   20     source frame (0 = a along the forward field, 1 = b)
//   19-0   source - target position, 10 bits per axis, biased by 512
// Everything but the position is known per frame pair (`splat.src`).
#define SPLAT_EMPTY 0xFFFFFFFFu
#define SPLAT_HIDDEN (1u << 29)

static struct pl_buffer_var uint_array_var(const char *name, int dim_v, size_t n)
{
    return (struct pl_buffer_var) {
        .var = { .name = name, .type = PL_VAR_UINT, .dim_v = dim_v, .dim_m = 1,
                 .dim_a = n },
        .layout = { .offset = 0, .stride = 4 * dim_v, .size = n * 4 * dim_v },
    };
}

// Prepare the splat of the current frame pair: find the layer order, then
// pack the vector and the key prefix of every pixel of both fields.
static bool splat_prepare(struct sh_motion_obj *obj, pl_gpu gpu, pl_dispatch dp)
{
    pl_tex fields[2] = { obj->flow.dense_fwd, obj->flow.dense_bwd };
    const int w = fields[0]->params.w, h = fields[0]->params.h;
    const size_t n = (size_t) w * h;
    struct pl_buffer_var mark_var = uint_array_var("splat_mark", 1, 2 * n);
    struct pl_buffer_var src_var = uint_array_var("splat_src", 2, 2 * n);

    if (!obj->splat.src || obj->splat.src->params.size != src_var.layout.size) {
        struct pl_buffer_var key0 = uint_array_var("splat_key0", 1, n);
        struct pl_buffer_var key1 = uint_array_var("splat_key1", 1, n);
        bool ok = pl_buf_recreate(gpu, &obj->splat.src, pl_buf_params(
                      .size = src_var.layout.size, .storable = true));
        ok &= pl_buf_recreate(gpu, &obj->splat.mark, pl_buf_params(
                      .size = mark_var.layout.size, .storable = true));
        for (int i = 0; i < 2; i++) {
            ok &= pl_buf_recreate(gpu, &obj->splat.key[i], pl_buf_params(
                      .size = key0.layout.size, .storable = true));
        }
        if (!ok) {
            PL_ERR(gpu, "pl_shader_frame_interp: failed creating splat buffers!");
            return false;
        }

        // The marks are compared against the pair counter, starting at 1, and
        // each splat clears the keys for the next one.
        obj->splat.pair = obj->splat.frame = 0;
        pl_shader sh = pl_dispatch_begin(dp);
        sh_describe(sh, "splat buffer initialization");
        sh->output = PL_SHADER_SIG_NONE;
        if (!sh_try_compute(sh, 16, 16, false, 0)) {
            pl_dispatch_abort(dp, &sh);
            return false;
        }

        bind_buf(sh, "splat_mark_buf", obj->splat.mark, PL_DESC_ACCESS_WRITEONLY, &mark_var, 1);
        bind_buf(sh, "splat_key0_buf", obj->splat.key[0], PL_DESC_ACCESS_WRITEONLY, &key0, 1);
        bind_buf(sh, "splat_key1_buf", obj->splat.key[1], PL_DESC_ACCESS_WRITEONLY, &key1, 1);
        // clang-format off
#pragma GLSL /* splat_prepare: initialization */                                \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(pos, size))) {                                             \
        int idx = pos.y * size.x + pos.x;                                       \
        splat_mark[idx] = splat_mark[size.x * size.y + idx] = 0u;               \
        splat_key0[idx] = splat_key1[idx] = ${const uint: SPLAT_EMPTY};         \
    }
        // clang-format on
        if (!pl_dispatch_compute(dp, pl_dispatch_compute_params(
                .shader = &sh,
                .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 })))
            return false;
    }

    // Layer order. A hidden pixel lands on the pixel covering it in the other
    // frame, which is thus in front, unless that one's vector leads straight
    // back (the two just disagree by a pixel, as on anti-aliased edges). The
    // same goes for the pixel the covering one came from in this frame, if
    // that one moves along with it.
    const unsigned stamp = ++obj->splat.pair;
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "splat layer order");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    ident_t fwd = bind_tex(sh, "flow_fwd", fields[0]);
    ident_t bwd = bind_tex(sh, "flow_bwd", fields[1]);
    bind_buf(sh, "splat_mark_buf", obj->splat.mark, PL_DESC_ACCESS_WRITEONLY, &mark_var, 1);
    const int spread = dc_env("X_SPREAD", 0);
    // clang-format off
#pragma GLSLH /* splat_prepare: layer order */                                  \
    /* mark p of field g and the pixels around it that move with it */          \
    void splat_stamp(int g, ivec2 p, vec2 v, uint stamp) {                      \
        ivec2 size = ivec2(${const int: w}, ${const int: h});                   \
        const int R = ${const int: spread};                                     \
        for (int dy = -R; dy <= R; dy++) {                                      \
            for (int dx = -R; dx <= R; dx++) {                                  \
                ivec2 np = p + ivec2(dx, dy);                                   \
                if (any(lessThan(np, ivec2(0))) || any(greaterThanEqual(np, size)))\
                    continue;                                                   \
                if (dx != 0 || dy != 0) {                                       \
                    vec2 nv = g == 0 ? texelFetch($fwd, np, 0).xy : texelFetch($bwd, np, 0).xy;\
                    if (length(nv - v) >= 0.3)                                  \
                        continue;                                               \
                }                                                               \
                splat_mark[(g * size.y + np.y) * size.x + np.x] = stamp;        \
            }                                                                   \
        }                                                                       \
    }
#pragma GLSL /* splat_prepare: layer order */                                   \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    int num    = size.x * size.y;                                               \
    for (int d = 0; d < 2; d++) {                                               \
        if (any(greaterThanEqual(pos, size)))                                   \
            break;                                                              \
        vec4 f = d == 0 ? texelFetch($fwd, pos, 0) : texelFetch($bwd, pos, 0);  \
        ivec2 q = ivec2(floor(vec2(pos) + f.xy + 0.5));                         \
        if (f.z >= 0.5 || any(lessThan(q, ivec2(0))) ||                         \
            any(greaterThanEqual(q, size)))                                     \
            continue;                                                           \
        vec2 o = d == 0 ? texelFetch($bwd, q, 0).xy : texelFetch($fwd, q, 0).xy;\
        if (length(o + f.xy) <= 0.5)                                            \
            continue;                                                           \
        splat_stamp(1 - d, q, o, ${uint: stamp});                               \
        ivec2 r = ivec2(floor(vec2(q) + o + 0.5));                              \
        if (any(lessThan(r, ivec2(0))) || any(greaterThanEqual(r, size)))       \
            continue;                                                           \
        vec2 fr = d == 0 ? texelFetch($fwd, r, 0).xy : texelFetch($bwd, r, 0).xy;\
        if (length(fr + o) < 0.5)                                               \
            splat_stamp(d, r, fr, ${uint: stamp});                              \
    }
    // clang-format on
    if (!pl_dispatch_compute(dp, pl_dispatch_compute_params(
            .shader = &sh,
            .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 })))
        return false;

    sh = pl_dispatch_begin(dp);
    const int xconf = dc_env("X_CONF", 1);
    const float expl_d = b_envf("X_EXPLD", 0.5f), expl_t = b_envf("X_EXPLT", 0.5f);
    const float rel_t = b_envf("X_RT", 1.0f), rel_n = b_envf("X_RN", 6.0f);
    const float rel_c = b_envf("X_RC", 32.0f), rel_h = b_envf("X_RH", 0.5f);
    const int rel_off = dc_env("X_ROFF", 0);
    sh_describe(sh, "splat source packing");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    fwd = bind_tex(sh, "flow_fwd", fields[0]);
    bwd = bind_tex(sh, "flow_bwd", fields[1]);
    bind_buf(sh, "splat_mark_buf", obj->splat.mark, PL_DESC_ACCESS_READONLY, &mark_var, 1);
    bind_buf(sh, "splat_src_buf", obj->splat.src, PL_DESC_ACCESS_WRITEONLY, &src_var, 1);
    // clang-format off
#pragma GLSL /* splat_prepare: packing */                                       \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    int num    = size.x * size.y;                                               \
    for (int d = 0; d < 2; d++) {                                               \
        if (any(greaterThanEqual(pos, size)))                                   \
            break;                                                              \
        vec4 f = d == 0 ? texelFetch($fwd, pos, 0) : texelFetch($bwd, pos, 0);  \
        bool hidden = f.z < 0.5;                                                \
        uint cost = uint(min(f.w * ${const float: MOTION_SPLAT_COST}, 254.0));  \
        if (hidden) {                                                           \
            /* Only trust this if the pixel lands on a visible one; the */      \
            /* vector is unreliable otherwise, which says nothing about */      \
            /* the visibility. */                                               \
            ivec2 q = ivec2(floor(vec2(pos) + f.xy + 0.5));                     \
            bool inside = all(greaterThanEqual(q, ivec2(0))) &&                 \
                          all(lessThan(q, size));                               \
            float vis = 0.0;                                                    \
            vec3 o = vec3(0.0);                                                 \
            if (inside) {                                                       \
                o = d == 0 ? texelFetch($bwd, q, 0).xyz : texelFetch($fwd, q, 0).xyz;\
                vis = o.z;                                                      \
            }                                                                   \
            hidden = vis >= 0.5 || ${const int: xconf} == 0;                    \
            if (hidden) {                                                       \
                /* Explained, if what covers it moves elsewhere and comes */    \
                /* from a pixel of this frame that agrees with that */          \
                bool explained = false;                                         \
                ivec2 r = ivec2(floor(vec2(q) + o.xy + 0.5));                   \
                if (inside && length(o.xy + f.xy) > ${const float: expl_d} &&   \
                    all(greaterThanEqual(r, ivec2(0))) && all(lessThan(r, size))) {\
                    vec3 fr = d == 0 ? texelFetch($fwd, r, 0).xyz : texelFetch($bwd, r, 0).xyz;\
                    explained = length(fr.xy + o.xy) < ${const float: expl_t} && fr.z >= 0.5;\
                }                                                               \
                cost = ${const uint: MOTION_SPLAT_COST_HIDDEN} + (explained ? 0u : 1u);\
            } else if (inside) {                                                \
                cost = 255u;                                                    \
            }                                                                   \
        }                                                                       \
                                                                                \
        int idx = d * num + pos.y * size.x + pos.x;                             \
        uint layer = splat_mark[idx] == ${uint: stamp} ? 1u : 0u;               \
        uint prefix = (3u - layer) << 30 | cost << 21;                          \
        if (hidden)                                                             \
            prefix |= ${const uint: SPLAT_HIDDEN};                              \
        /* EXPERIMENT: reliability of the vector, bits 0-7 */                   \
        float coh = 0.0;                                                        \
        for (int j = 0; j < 9; j++) {                                           \
            ivec2 np = clamp(pos + ivec2(j % 3, j / 3) - 1, ivec2(0), size - 1);\
            vec2 nv = d == 0 ? texelFetch($fwd, np, 0).xy : texelFetch($bwd, np, 0).xy;\
            coh += length(nv - f.xy) < ${float: rel_t} ? 1.0 : 0.0;             \
        }                                                                       \
        coh = clamp((coh - 1.0) / ${float: rel_n}, 0.0, 1.0);                   \
        float rel = coh * exp(-float(cost) / ${float: rel_c})                    \
                  * (hidden ? ${float: rel_h} : 1.0);                           \
        if (${const int: rel_off} != 0) rel = 1.0;                              \
        prefix |= uint(clamp(rel, 0.0, 1.0) * 255.0 + 0.5);                     \
        splat_src[idx] = uvec2(packHalf2x16(f.xy), prefix);                     \
    }
    // clang-format on
    if (!pl_dispatch_compute(dp, pl_dispatch_compute_params(
            .shader = &sh,
            .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 })))
        return false;

    obj->splat.dirty = false;
    return true;
}

// Splat both fields to the time `t` between the frames, into
// `splat.key[splat.frame & 1]`. Clears the other key buffer for the next call
// along the way.
static bool splat_scatter(struct sh_motion_obj *obj, pl_dispatch dp, float t)
{
    const int w = obj->flow.dense_fwd->params.w, h = obj->flow.dense_fwd->params.h;
    const size_t n = (size_t) w * h;
    const int cur = ++obj->splat.frame & 1;
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "splat");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    struct pl_buffer_var src_var = uint_array_var("splat_src", 2, 2 * n);
    struct pl_buffer_var key_var = uint_array_var("splat_key", 1, n);
    struct pl_buffer_var next_var = uint_array_var("splat_next", 1, n);
    bind_buf(sh, "splat_src_buf", obj->splat.src, PL_DESC_ACCESS_READONLY, &src_var, 1);
    bind_buf(sh, "splat_key_buf", obj->splat.key[cur], PL_DESC_ACCESS_READWRITE, &key_var, 1);
    bind_buf(sh, "splat_next_buf", obj->splat.key[!cur], PL_DESC_ACCESS_WRITEONLY, &next_var, 1);
    // clang-format off
#pragma GLSL /* splat_scatter */                                                \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    int idx    = pos.y * size.x + pos.x;                                        \
    float t    = ${float: t};                                                   \
    if (all(lessThan(pos, size)))                                               \
        splat_next[idx] = ${const uint: SPLAT_EMPTY};                           \
    for (int d = 0; d < 2; d++) {                                               \
        if (any(greaterThanEqual(pos, size)))                                   \
            break;                                                              \
        uvec2 src = splat_src[d * size.x * size.y + idx];                       \
        vec2 f = unpackHalf2x16(src.x);                                         \
        ivec2 q = ivec2(floor(vec2(pos) + (d == 0 ? t : 1.0 - t) * f + 0.5));   \
        ivec2 off = pos - q;                                                    \
        if (any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size)) ||     \
            any(greaterThan(abs(off), ivec2(511))))                             \
            continue;                                                           \
        uint key = (src.y & 0xFFE00000u) | uint(d) << 20 | uint(off.y + 512) << 10\
                 | uint(off.x + 512);                                           \
        atomicMin(splat_key[q.y * size.x + q.x], key);                          \
    }
    // clang-format on
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));
}

// Two contiguous histograms describing the global motion in the frame.
// One histogram for each component of the motion vector (x,y).
static const struct pl_buffer_var global_hist_var = {
    .var = {
        .name  = "global_hist",
        .type  = PL_VAR_UINT,
        .dim_v = 1,
        .dim_m = 1,
        .dim_a = 2 * MOTION_GLOBAL_BINS,
    },
    .layout = {
        .offset = 0,
        .size   = 2 * MOTION_GLOBAL_BINS * sizeof(uint32_t),
        .stride = sizeof(uint32_t),
    },
};

// Reusable helper to recreate global motion and histogram scratch buffers.
static bool global_buf_recreate(pl_gpu gpu, pl_buf *buf, size_t size,
                                bool host_readable)
{
    if (*buf)
        return true;

    // .initial_data doesn't care if it is larger than .size
    static const uint32_t zero[2 * MOTION_GLOBAL_BINS] = { 0 };
    pl_assert(size <= sizeof(zero));
    *buf = pl_buf_create(
        gpu, pl_buf_params(.size = size, .storable = true,
                           .host_readable = host_readable,
                           .memory_type = host_readable ? PL_BUF_MEM_AUTO
                                                        : PL_BUF_MEM_DEVICE,
                           .initial_data = zero, .debug_tag = PL_DEBUG_TAG, ));

    if (!*buf) {
        PL_ERR(gpu,
               "pl_motion_estimate: failed creating %zu byte global "
               "motion buffer!",
               size);
        return false;
    }

    return true;
}

// Create a histogram of motion from the blocks of `flow` into `hist`.
// NOTE: if the median motion is outside of [-`MOTION_GLOBAL_BINS/2`,
// `MOTION_GLOBAL_BINS-1`], the global motion is clamped.
static bool motion_global_hist_pass(pl_gpu gpu, pl_dispatch dispatch,
                                    pl_tex flow, pl_buf hist)
{
    const int grid_w = flow->params.w, grid_h = flow->params.h;
    const int dim = MOTION_GLOBAL_WORKGROUP_DIM;
    const size_t shmem = 2 * MOTION_GLOBAL_BINS * sizeof(uint32_t);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "global motion histogram (%dx%d blocks)", grid_w, grid_h);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, dim, dim, false, shmem)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create global motion histogram "
               "shader, needs %dx%d threads and %zu bytes of shared memory "
               "per work group!",
               dim, dim, shmem);
        goto error;
    }

    ident_t src = bind_tex(sh, "flow_in", flow);
    bind_buf(sh, "global_hist_buf", hist, PL_DESC_ACCESS_READWRITE,
             &global_hist_var, 1);

    // clang-format off
#pragma GLSLH /* motion_global_hist_pass */                                     \
    shared uint s_hist[2][${const int: MOTION_GLOBAL_BINS}];

#pragma GLSL /* motion_global_hist_pass */                                      \
    /* One thread per bin. */                                                   \
    int tid = int(gl_LocalInvocationIndex);                                     \
    s_hist[0][tid] = s_hist[1][tid] = 0u; /* zero shared memory */              \
    barrier();                                                                  \
                                                                                \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 blk = ivec2(gl_GlobalInvocationID.xy);                                \
    if (all(lessThan(blk, grid))) {                                             \
        vec2 mv = texelFetch($src, blk, 0).xy;                                  \
        /* Shift mv in [-128, 128] to bin index in [0, 256). */                 \
        ivec2 bin = ivec2(round(mv)) + ${const int: MOTION_GLOBAL_BINS / 2};    \
        bin = clamp(bin, ivec2(0), ivec2(${const int: MOTION_GLOBAL_BINS - 1}));\
        atomicAdd(s_hist[0][bin.x], 1u);                                        \
        atomicAdd(s_hist[1][bin.y], 1u);                                        \
    }                                                                           \
    barrier();                                                                  \
                                                                                \
    for (int c = 0; c < 2; c++) {                                               \
        if (s_hist[c][tid] != 0u) /* prefer branching over unnecessary atomic */\
            atomicAdd(global_hist[c * ${const int: MOTION_GLOBAL_BINS} + tid],  \
                      s_hist[c][tid]);                                          \
    }
    // clang-format on

    const struct pl_dispatch_compute_params *params = pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = {
            PL_DIV_UP(grid_w, dim),
            PL_DIV_UP(grid_h, dim),
            1,
        },
    );

    return pl_dispatch_compute(dispatch, params);

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Reads the median from the histogram. Needs to be in a separate shader because
// the shader that builds the histogram uses many blocks.
// TODO: estimate a confidence. It is always 1 for now. One approach would be
// using the proportion of bins in the histogram that are close to the median.
static bool motion_global_median_pass(pl_gpu gpu, pl_dispatch dispatch,
                                      pl_buf hist, int num, pl_buf out,
                                      enum pl_motion_direction dir)
{
    const size_t shmem = 2 * MOTION_GLOBAL_BINS * sizeof(uint32_t);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "global motion median (%d blocks)", num);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, MOTION_GLOBAL_BINS, 1, false, shmem)) {
        PL_ERR(gpu,
               "pl_motion_estimate: cannot create global motion median "
               "shader, needs %d threads and %zu bytes of shared memory per "
               "work group!",
               MOTION_GLOBAL_BINS, shmem);
        goto error;
    }

    bind_buf(sh, "global_hist_buf", hist, PL_DESC_ACCESS_READWRITE,
             &global_hist_var, 1);
    bind_buf(sh, "global_motion_out", out, PL_DESC_ACCESS_WRITEONLY,
             pl_motion_global_buf_vars[dir], PL_MOTION_GLOBAL_BUF_NUM_VARS);

    // clang-format off
#pragma GLSLH /* motion_global_median_pass */                                   \
    shared uint s_hist[2][${const int: MOTION_GLOBAL_BINS}];

#pragma GLSL /* motion_global_median_pass */                                    \
    /* One thread per bin for loading and clearing the histogram. */            \
    int tid = int(gl_LocalInvocationIndex);                                     \
    for (int c = 0; c < 2; c++) {                                               \
        int g = c * ${const int: MOTION_GLOBAL_BINS} + tid;                     \
        s_hist[c][tid] = global_hist[g];                                        \
        global_hist[g] = 0u;                                                    \
    }                                                                           \
    barrier();                                                                  \
                                                                                \
    /* Single-threaded prefix sum to find the median. */                        \
    if (tid < 2) {                                                              \
        uint sum = 0u;                                                          \
        int i = 0;                                                              \
        for (; i < ${const int: MOTION_GLOBAL_BINS}; i++) {                     \
            sum += s_hist[tid][i];                                              \
            if (sum > ${const uint: num / 2})                                   \
                break;                                                          \
        }                                                                       \
        /* Shift bin index in [0, 256) to mv in [-128, 128). */                 \
        float mv = float(i - ${const int: MOTION_GLOBAL_BINS / 2});             \
        @if (dir == PL_MOTION_FORWARD) {                                        \
            global_mv_fwd[tid] = mv;                                            \
            if (tid == 0)                                                       \
                global_confidence_fwd = 1.0;                                    \
        @} else {                                                               \
            global_mv_bwd[tid] = mv;                                            \
            if (tid == 0)                                                       \
                global_confidence_bwd = 1.0;                                    \
        @}                                                                      \
    }
    // clang-format on

    const struct pl_dispatch_compute_params *params = pl_dispatch_compute_params(
        .dispatch_size = { 1, 1, 1 },
        .shader = &sh,
    );

    return pl_dispatch_compute(dispatch, params);

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Estimates the global motion of `dir` from its flow field.
static bool estimate_global(pl_gpu gpu, pl_dispatch dispatch,
                            struct sh_motion_obj *obj,
                            enum pl_motion_direction dir)
{
    const bool fwd = dir == PL_MOTION_FORWARD;
    pl_tex flow = fwd ? obj->flow.fwd.elem[0] : obj->flow.bwd.elem[0];
    pl_buf *global = fwd ? &obj->global.fwd : &obj->global.bwd;

    return global_buf_recreate(gpu, &obj->global.hist,
                               2 * MOTION_GLOBAL_BINS * sizeof(uint32_t),
                               false) &&
           global_buf_recreate(gpu, global, sizeof(struct global_buf_data),
                               true) &&
           motion_global_hist_pass(gpu, dispatch, flow, obj->global.hist) &&
           motion_global_median_pass(gpu, dispatch, obj->global.hist,
                                     flow->params.w * flow->params.h, *global,
                                     dir);
}

// Stage B after the per pixel selection: occlusion, then fill
static bool motion_b_post(pl_gpu gpu, pl_dispatch dispatch,
                          struct sh_motion_obj *obj, pl_tex luma_a, pl_tex luma_b)
{
    pl_tex luma[2] = { luma_a, luma_b };
    pl_tex dense[2] = { obj->flow.dense_fwd, obj->flow.dense_bwd };
    pl_tex sel[2] = { obj->b_sel[0], obj->b_sel[1] };
    for (int d = 0; d < 2 && dc_env("LM_B_SHARP", 0); d++) {
        if (!dense_flow_recreate(gpu, &obj->b_shp[d], obj->w, obj->h,
                                 obj->b_sel[d]->params.format) ||
            !motion_sharpen_pass(gpu, dispatch, luma[d], luma[!d], obj->b_sel[d],
                                 obj->b_shp[d]))
            return false;
        sel[d] = obj->b_shp[d];
    }
    for (int d = 0; d < 2; d++) {
        if (!motion_occlusion_pass(gpu, dispatch, luma[d], sel[d],
                                   sel[!d], obj->b_occ[d]) ||
            !motion_fill_pass(gpu, dispatch, luma[d], luma[!d], obj->b_occ[d],
                              dense[d]))
            return false;
    }
    return true;
}

bool pl_motion_estimate(const struct pl_motion_estimate_input *input,
                        const struct pl_motion_estimate_params *params)
{
    if (!input || !input->gpu || !input->dispatch || !input->state)
        return false;

    params = PL_DEF(params, &pl_motion_estimate_default_params);

    pl_gpu gpu = input->gpu;
    struct sh_motion_obj *obj =
        sh_alloc_obj(gpu, input->state, PL_SHADER_OBJ_MOTION,
                     sizeof(struct sh_motion_obj), motion_uninit);
    if (!obj)
        return false;

    if (!validate(gpu, input, params)) {
        invalidate_temporal(gpu, obj, "invalid input");
        return false;
    }

    const int w = input->a.tex->params.w;
    const int h = input->a.tex->params.h;
    const char *changed = NULL;
    if (w != obj->w || h != obj->h) {
        changed = "input dimensions changed";
    } else if (!params_eq(&obj->params, params)) {
        changed = "parameters changed";
    }

    // EXPERIMENT: stage B needs both directions
    const int bmode = input->direction == PL_MOTION_BOTH
                    ? dc_env("LM_B", motion_b_request) : 0;
    if (bmode != obj->b_mode) {
        changed = "parameters changed";
        obj->b_mode = bmode;
        obj->sig_a = obj->sig_b = 0;
    }

    if (changed) {
        invalidate_temporal(gpu, obj, changed);
        obj->params = *params;
        obj->w = w;
        obj->h = h;
    }

    if (obj->sig_a == input->a.sig && obj->sig_b == input->b.sig &&
        (obj->flow.valid & input->direction) == input->direction)
    {
        // Same frame pair as the previous call, at most the global motion is
        // left to compute.
        if (!input->estimate_global || obj->global.valid)
            return true;

        for (int i = 0; i < 2; i++) {
            const enum pl_motion_direction dir = 1 << i;
            if ((obj->flow.valid & dir) &&
                !estimate_global(gpu, input->dispatch, obj, dir))
                goto error;
        }

        obj->global.valid = true;
        return true;
    }

    if (obj->sig_b != input->a.sig) {
        invalidate_temporal(
            gpu, obj,
            "frame signature of first frame not equal to frame signature "
            "of second frame from previous call");
    }

    obj->sig_a = input->a.sig;
    obj->sig_b = input->b.sig;
    obj->splat.dirty = true;

    const int levels = pyramid_num_levels(w, h, params);
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, 1, 16, 0,
                             PL_FMT_CAP_LINEAR | PL_FMT_CAP_RENDERABLE);
    if (!fmt) {
        PL_ERR(gpu, "pl_motion_estimate: no renderable, linearly filterable "
                    "single component float format available!");
        goto error;
    }

    // The search covers the trailing partial blocks as well, hence ceiling
    // division.
    obj->grid_w = PL_DIV_UP(w, params->block_size);
    obj->grid_h = PL_DIV_UP(h, params->block_size);

    pl_fmt mv_fmt = flow_fmt(gpu);
    if (!mv_fmt) {
        PL_ERR(gpu, "pl_motion_estimate: no storable, linearly filterable "
                    "two component float format available!");
        goto error;
    }

    // These are no-ops as long as the motion search shape stays the same.
    // Otherwise, they invalidate the no longer usable temporal prior that resides
    // in the `obj->flow.fwd` and `obj->flow.bwd`.
    if (input->direction & PL_MOTION_FORWARD) {
        if (!flow_pyramid_recreate(gpu, obj, &obj->flow.fwd, w, h, levels,
                                   params->block_size, mv_fmt) ||
            !dense_flow_recreate(gpu, &obj->flow.dense_fwd, w, h, mv_fmt))
            goto error;
    }

    if (input->direction & PL_MOTION_BACKWARD) {
        if (!flow_pyramid_recreate(gpu, obj, &obj->flow.bwd, w, h, levels,
                                   params->block_size, mv_fmt) ||
            !dense_flow_recreate(gpu, &obj->flow.dense_bwd, w, h, mv_fmt))
            goto error;
    }

    for (int i = 0; i < 2 && bmode; i++) {
        if (!dense_flow_recreate(gpu, &obj->b_sel[i], w, h, mv_fmt) ||
            !dense_flow_recreate(gpu, &obj->b_occ[i], w, h, mv_fmt))
            goto error;
    }

    if (!flow_pyramid_recreate(gpu, obj, &obj->scratch.raw, w, h, levels,
                               params->block_size, mv_fmt))
        goto error;

    if (!flow_pyramid_recreate(gpu, obj, &obj->scratch.filtered, w, h, levels,
                               params->block_size, mv_fmt))
        goto error;

    // Upon cold start or after reset, build the image pyramid for frame `a`,
    // use cached pyramid otherwise.
    if (!obj->pyramid_valid) {
        if (!pyramid_recreate(gpu, obj, &obj->pyramid_prev, w, h, levels, fmt))
            goto error;
        if (!convert_to_luma(input->dispatch, &input->a,
                             obj->pyramid_prev.elem[0]))
        {
            PL_ERR(gpu, "pl_motion_estimate: failed converting frame a to "
                        "luma!");
            goto error;
        }
        if (!pyramid_build(gpu, input->dispatch, &obj->pyramid_prev))
            goto error;
    }

    if (!pyramid_recreate(gpu, obj, &obj->scratch.levels, w, h, levels, fmt))
        goto error;
    if (!convert_to_luma(input->dispatch, &input->b,
                         obj->scratch.levels.elem[0]))
    {
        PL_ERR(gpu, "pl_motion_estimate: failed converting frame b to luma!");
        goto error;
    }
    if (!pyramid_build(gpu, input->dispatch, &obj->scratch.levels))
        goto error;

    static const enum pl_motion_direction dirs[] = {
        PL_MOTION_FORWARD,
        PL_MOTION_BACKWARD,
    };

    void *tmp = pl_tmp(NULL);
    bool ok = true;
    for (int d = 0; d < PL_ARRAY_SIZE(dirs); d++) {
        if (!(input->direction & dirs[d]))
            continue;

        const bool fwd = dirs[d] == PL_MOTION_FORWARD;
        tex_pyramid_t *field = fwd ? &obj->flow.fwd : &obj->flow.bwd;

        const bool has_prior = obj->flow.valid & dirs[d];

        for (int i = levels - 1; i >= 0; i--) {
            const bool coarsest = levels > 1 && i == levels - 1;
            const int radius = coarsest ? params->search_radius_coarse
                                        : params->search_radius_fine;

            pl_tex raw = obj->scratch.raw.elem[i];
            pl_tex out = obj->scratch.filtered.elem[i];

            pl_tex a = obj->pyramid_prev.elem[i];
            pl_tex b = obj->scratch.levels.elem[i];

            // Use the parent motion search to seed this levels search, for
            // all levels but the coarsest. The parent is the motion field
            // after smoothing.
            pl_tex parent =
                i < levels - 1 ? obj->scratch.filtered.elem[i + 1] : NULL;

            // Seed the motion search using the motion estimate at the
            // corresponding level of the motion search pyramid from the
            // previous frame pair that the motion estimation was invoked
            // upon.
            pl_tex prior = has_prior && !getenv("A_NOPRIOR") ? field->elem[i] : NULL;

            struct motion_geometry geom;
            ok = motion_plan(gpu, tmp, params->block_size, radius,
                             motion_seeds_plan(parent != NULL, prior != NULL),
                             i == 0 && !getenv("A_NOSUB"), out->params.w, out->params.h, &geom);
            if (!ok)
                break;

            const struct motion_search search = {
                .cand_lut = &obj->cand_lut[coarsest],
                .ref    = fwd ? a : b,
                .tgt    = fwd ? b : a,
                .parent = parent,
                .prior  = prior,
                .out_mv = raw,
                .geom   = &geom,
            };

            ok = motion_search_pass(gpu, input->dispatch, &search);
            if (!ok)
                break;

            const int nfine = getenv("FINE") ? atoi(getenv("FINE")) : 0;
            pl_tex dbg = NULL;
            if (i == 0 && !nfine) {
                pl_tex_recreate(gpu, &obj->raw_dbg[d], &raw->params);
                dbg = obj->raw_dbg[d];
            }
            ok = motion_smooth_pass(gpu, input->dispatch, params->block_size,
                                    fwd ? a : b, fwd ? b : a, raw, out,
                                    i == 0, dbg);
            if (!ok)
                break;
        }

        if (!ok)
            break;

        // EXPERIMENT: extra full resolution stages with half the block size
        const int nfine = getenv("FINE") ? atoi(getenv("FINE")) : 0;
        pl_tex prev = obj->scratch.filtered.elem[0];
        for (int s = 1; ok && s <= nfine && s < 4; s++) {
            const int fbs = params->block_size >> s;
            pl_tex a = obj->pyramid_prev.elem[0];
            pl_tex b = obj->scratch.levels.elem[0];
            struct pl_tex_params tp = prev->params;
            tp.w = PL_DIV_UP(a->params.w, fbs);
            tp.h = PL_DIV_UP(a->params.h, fbs);
            tp.debug_tag = PL_DEBUG_TAG;
            ok = pl_tex_recreate(gpu, &obj->fine_raw[s], &tp) &&
                 pl_tex_recreate(gpu, &obj->fine[d][s], &tp);
            pl_tex dbg = NULL;
            if (ok && s == nfine) {
                ok = pl_tex_recreate(gpu, &obj->raw_dbg[d], &tp);
                dbg = obj->raw_dbg[d];
            }
            struct motion_geometry geom;
            ok = ok && motion_plan(gpu, tmp, fbs, params->search_radius_fine,
                                   motion_seeds_plan(true, false), true,
                                   tp.w, tp.h, &geom);
            if (!ok)
                break;
            const struct motion_search search = {
                .cand_lut = &obj->cand_lut[0],
                .ref    = fwd ? a : b,
                .tgt    = fwd ? b : a,
                .parent = prev,
                .out_mv = obj->fine_raw[s],
                .geom   = &geom,
                .parent_scale = 1,
            };
            ok = motion_search_pass(gpu, input->dispatch, &search) &&
                 motion_smooth_pass(gpu, input->dispatch, fbs, fwd ? a : b,
                                    fwd ? b : a, obj->fine_raw[s],
                                    obj->fine[d][s], true, dbg);
            prev = obj->fine[d][s];
        }
        const int two = bmode ? dc_env("LM_B_TWO", 0) : 0;
        if (two) {
            const int pbs = params->block_size >> PL_MIN(nfine, 3);
            pl_tex ra = fwd ? obj->pyramid_prev.elem[0] : obj->scratch.levels.elem[0];
            pl_tex rb = fwd ? obj->scratch.levels.elem[0] : obj->pyramid_prev.elem[0];
            ok = ok && dense_flow_recreate(gpu, &obj->b_sub[d], PL_DIV_UP(ra->params.w, two),
                                           PL_DIV_UP(ra->params.h, two), prev->params.format) &&
                 motion_subsel_pass(gpu, input->dispatch, pbs, two, ra, rb, prev, obj->b_sub[d],
                                    has_prior && dc_env("LM_B2_TEMP", 1) ?
                                    (fwd ? obj->flow.dense_fwd : obj->flow.dense_bwd) : NULL) &&
                 (dc_env("LM_B2_NOREF", 0)
                  ? motion_upsample_pass(gpu, input->dispatch, two, obj->b_sub[d], obj->b_sel[d])
                  : motion_refine_pass(gpu, input->dispatch, two, ra, rb, obj->b_sub[d],
                                       obj->b_sel[d], true));
        } else if (bmode && dc_env("LM_B_SEL", 1)) {
            ok = ok && motion_select_pass(gpu, input->dispatch,
                                params->block_size >> PL_MIN(nfine, 3),
                                fwd ? obj->pyramid_prev.elem[0]
                                    : obj->scratch.levels.elem[0],
                                fwd ? obj->scratch.levels.elem[0]
                                    : obj->pyramid_prev.elem[0],
                                prev, obj->b_sel[d],
                                has_prior ? (fwd ? obj->flow.dense_fwd
                                                 : obj->flow.dense_bwd) : NULL);
        } else {
            ok = ok && motion_refine_pass(gpu, input->dispatch,
                                params->block_size >> PL_MIN(nfine, 3),
                                fwd ? obj->pyramid_prev.elem[0]
                                    : obj->scratch.levels.elem[0],
                                fwd ? obj->scratch.levels.elem[0]
                                    : obj->pyramid_prev.elem[0],
                                prev,
                                bmode ? obj->b_sel[d] :
                                fwd ? obj->flow.dense_fwd : obj->flow.dense_bwd,
                                bmode);
        }
        if (!ok)
            break;

        // NOTE: we assume that the prior forward motion field is safe to be
        // overwritten by the backward motion search.
        //
        // NOTE: motion estimation that succeeds on the forward motion
        // estimation call but fails on the backward motion estimation call
        // may leave sh_motion_obj in an incomplete state. we assume that if
        // the backward direction fails, the caller is not interested in the
        // forward direction results.
        PL_SWAP(*field, obj->scratch.filtered);

        if (input->estimate_global) {
            ok = estimate_global(gpu, input->dispatch, obj, dirs[d]);
            if (!ok)
                break;
        }
    }
    pl_free(tmp);

    if (!ok)
        goto error;

    if (bmode && !motion_b_post(gpu, input->dispatch, obj, obj->pyramid_prev.elem[0],
                                obj->scratch.levels.elem[0]))
        goto error;

    obj->flow.valid = input->direction;
    obj->global.valid = input->estimate_global;

    // Cache frame `b`'s luma pyramid, use reuse `a` buffers as scratch.
    PL_SWAP(obj->pyramid_prev, obj->scratch.levels);
    obj->pyramid_valid = true;

    return true;

error:
    invalidate_temporal(gpu, obj, "estimation failed");
    return false;
}

// Mixes the two frames, each sampled at the position offset by the global
// motion, scaled by the frame's distance in time to the intermediate frame.
static bool frame_interp_global(pl_shader sh,
                                const struct pl_frame_interp_input *input,
                                const struct pl_color_space *out_color)
{
    pl_shader_obj state = *input->state;
    if (!pl_motion_get_global_buf(state, PL_MOTION_FORWARD)) {
        PL_ERR(sh, "pl_shader_frame_interp: no global motion estimate "
               "available!");
        return false;
    }

    const struct pl_tex_params *tpars = &input->a.tex->params;
    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;

    sh_describe(sh, "frame interpolation (global motion)");
    ident_t gm = sh_motion_global(sh, state, PL_MOTION_FORWARD);
    pl_assert(gm);

    const float t = PL_CLAMP(input->t, 0.0f, 1.0f);
    const struct pl_motion_estimate_frame *frames[2] = { &input->a, &input->b };

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
    vec4 color;                                                                 \
    {                                                                           \
    vec4 mix_color = vec4(0.0);                                                 \
    vec4 mix_clamped = vec4(0.0);  /* fallback for when global motion offset is \
                                      out of frame for both frames */           \
    float mix_inside = 0.0;  /* sum of weights of the pixels with valid global  \
                                motion offset */                                \
    vec2 gm_pos;  /* pixel position offset by global motion, scaled by t */     \
    float gm_inside;  /* 0 if gm_pos is out of bounds, 1 otherwise */
    // clang-format on

    for (int i = 0; i < 2; i++) {
        pl_tex tex = frames[i]->tex;
        enum pl_tex_sample_mode sample_mode = PL_TEX_SAMPLE_NEAREST;
        if (tex->params.format->caps & PL_FMT_CAP_LINEAR)
            sample_mode = PL_TEX_SAMPLE_LINEAR; // offsets are fractional

        ident_t pos, pt, frame = sh_bind(sh, tex, PL_TEX_ADDRESS_CLAMP,
                                         sample_mode, "frame", NULL, &pos, &pt);
        if (!frame)
            return false;

        // A pixel in the intermediate frame finds its corresponding pixel in
        // frame `a` at (x,y) - t*gm, and in frame `b` at (x,y) + (1-t)*gm.
        const float offset = i == 0 ? -t : 1.0f - t;
        ident_t weight = SH_FLOAT_DYN(i == 0 ? 1.0f - t : t);

        // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
        gm_pos = $pos + ${dynamic float: offset} * $gm.xy * $pt;                \
        gm_inside = all(greaterThanEqual(gm_pos, vec2(0.0))) &&                 \
                    all(lessThanEqual(gm_pos, vec2(1.0))) ? 1.0 : 0.0;          \
        color = textureLod($frame, gm_pos, 0.0);
        // clang-format on

        struct pl_color_repr repr = frames[i]->repr;
        struct pl_color_space csp = frames[i]->color;
        pl_color_space_infer(&csp);
        if (!pl_color_space_equal(&csp, out_color)) {
            pl_shader_set_alpha(sh, &repr, PL_ALPHA_INDEPENDENT);
            pl_shader_color_map_ex(sh, NULL, pl_color_map_args(csp, *out_color));
        }
        pl_shader_set_alpha(sh, &repr, PL_ALPHA_PREMULTIPLIED);

        // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
        mix_clamped += $weight * color;                                         \
        /* Discard pixels that are out of frame. */                             \
        mix_color += gm_inside * $weight * color;                               \
        mix_inside += gm_inside * $weight;
        // clang-format on
    }

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (global motion) */                       \
    /* Reweight contributions of pixels to only consider those that are in      \
     * frame. */                                                                \
    color = mix_inside > 0.0 ? mix_color / mix_inside : mix_clamped;            \
    }
    // clang-format on

    return true;
}

// Draws the forward flow with the Middlebury color wheel
static bool frame_interp_visualize(pl_shader sh,
                                   const struct pl_frame_interp_input *input,
                                   const struct pl_color_space *out_color)
{
    const struct pl_tex_params *tpars = &input->a.tex->params;
    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;

    ident_t flow = sh_motion_flow(sh, *input->state, PL_MOTION_FORWARD);
    if (!flow) {
        PL_ERR(sh, "pl_shader_frame_interp: no flow field available!");
        return false;
    }

    ident_t pos, frame = sh_bind(sh, input->a.tex, PL_TEX_ADDRESS_CLAMP,
                                 PL_TEX_SAMPLE_NEAREST, "frame", NULL, &pos,
                                 NULL);
    if (!frame)
        return false;

    sh_describe(sh, "frame interpolation (visualize motion)");

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (visualize motion) */                    \
    vec4 color;                                                                 \
    {                                                                           \
    color = textureLod($frame, $pos, 0.0);
    // clang-format on

    struct pl_color_repr repr = input->a.repr;
    struct pl_color_space frame_csp = input->a.color;
    pl_color_space_infer(&frame_csp);
    if (!pl_color_space_equal(&frame_csp, out_color)) {
        pl_shader_set_alpha(sh, &repr, PL_ALPHA_INDEPENDENT);
        pl_shader_color_map_ex(sh, NULL,
                               pl_color_map_args(frame_csp, *out_color));
    }
    pl_shader_set_alpha(sh, &repr, PL_ALPHA_PREMULTIPLIED);

    // The wheel is piecewise linear between red, yellow, green, cyan, blue and
    // magenta, with segment lengths 15, 6, 4, 11, 13, 6. As in the original
    // implementation, the hue spans only [0, 54], leaving the last segment
    // incomplete.
    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (visualize motion) */                    \
    vec4 frame_color = color;                                                   \
    const vec3 R = vec3(1.0, 0.0, 0.0), Y = vec3(1.0, 1.0, 0.0),                \
               G = vec3(0.0, 1.0, 0.0), C = vec3(0.0, 1.0, 1.0),                \
               B = vec3(0.0, 0.0, 1.0), M = vec3(1.0, 0.0, 1.0);                \
    vec3 mv = textureLod($flow, $pos, 0.0).xyz;                                 \
    vec2 uv = mv.xy;                                                            \
    float rad = length(uv);                                                     \
    vec3 col = vec3(1.0);                                                       \
    if (rad > 0.0) {                                                            \
        float fk = (atan(-uv.y, -uv.x) / ${const float: M_PI} + 1.0) * 27.0;    \
        if (fk < 15.0)      col = mix(R, Y, fk / 15.0);                         \
        else if (fk < 21.0) col = mix(Y, G, (fk - 15.0) / 6.0);                 \
        else if (fk < 25.0) col = mix(G, C, (fk - 21.0) / 4.0);                 \
        else if (fk < 36.0) col = mix(C, B, (fk - 25.0) / 11.0);                \
        else if (fk < 49.0) col = mix(B, M, (fk - 36.0) / 13.0);                \
        else                col = mix(M, R, (fk - 49.0) / 6.0);                 \
        col = vec3(1.0) - rad / (rad + 4.0) * (vec3(1.0) - col);                \
    }                                                                           \
    color = vec4(mv.z * col, 1.0);
    // clang-format on

    struct pl_color_space csp = pl_color_space_srgb;
    pl_color_space_infer(&csp);
    pl_shader_color_map_ex(sh, NULL, pl_color_map_args(csp, *out_color));

    // clang-format off
#pragma GLSL /* pl_shader_frame_interp (visualize motion) */                    \
    color = mix(frame_color, color, 0.95);                                      \
    }
    // clang-format on

    return true;
}

// --- EXPERIMENTAL: PL_FRAME_INTERP_LOCAL_MOTION ---------------------------
//
// Every output pixel is a mix of both frames compensated by a handful of
// candidate motion vectors: the global motion, and the per pixel flow of
// each direction at the four points of a coarse lattice around the pixel.
// Each candidate weighs in by how close and how confident its lattice point
// is, and by how well the two samples it fetches agree with each other,
// relative to the best candidate. Where the candidates agree on the motion,
// the frames are instead resampled once along their mean with a sharper
// filter. Whatever weight the candidates' confidence leaves unclaimed goes to
// plain frame mixing.
//
// The flow fields are anchored to their source frame, whereas a lattice point
// lies in the frame being synthesized. So the flow is not read at the point
// itself, but where its content came from, found by fixed point iteration.
//
// Tunables live in `pl_frame_interp_params.local`.

static bool frame_interp_local(pl_shader sh,
                               const struct pl_frame_interp_input *input,
                               const struct pl_frame_interp_params *params,
                               const struct pl_color_space *out_color)
{
    pl_tex flows[2] = {
        pl_motion_get_flow(*input->state, PL_MOTION_FORWARD),
        pl_motion_get_flow(*input->state, PL_MOTION_BACKWARD),
    };
    if (!flows[0]) {
        PL_ERR(sh, "pl_shader_frame_interp: no flow field available!");
        return false;
    }

    const struct pl_frame_interp_local_params *lm = &params->local;
    const float t = PL_CLAMP(input->t, 0.0f, 1.0f);
    const int ndir = lm->bidir && flows[1] ? 2 : 1;

    // Emit the color conversion of each frame as a function, so it can be
    // applied to every candidate's sample without duplicating it.
    const struct pl_motion_estimate_frame *frames[2] = { &input->a, &input->b };
    ident_t conv[2];
    for (int i = 0; i < 2; i++) {
        pl_shader sub = pl_dispatch_begin_ex(input->dispatch, true);
        if (sub->prefix == sh->prefix) {
            // The first unique identifier may coincide with `sh`'s, which
            // `sh_subpass` rejects.
            pl_dispatch_abort(input->dispatch, &sub);
            sub = pl_dispatch_begin_ex(input->dispatch, true);
        }
        sh_require(sub, PL_SHADER_SIG_COLOR, 0, 0);
        struct pl_color_repr repr = frames[i]->repr;
        struct pl_color_space csp = frames[i]->color;
        pl_color_space_infer(&csp);
        if (!pl_color_space_equal(&csp, out_color)) {
            pl_shader_set_alpha(sub, &repr, PL_ALPHA_INDEPENDENT);
            pl_shader_color_map_ex(sub, NULL, pl_color_map_args(csp, *out_color));
        }
        pl_shader_set_alpha(sub, &repr, PL_ALPHA_PREMULTIPLIED);
        conv[i] = sh_subpass(sh, sub);
        pl_dispatch_abort(input->dispatch, &sub);
        if (!conv[i])
            return false;
    }

    const struct pl_tex_params *tpars = &input->a.tex->params;
    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;
    sh_describe(sh, "frame interpolation (local motion)");

    ident_t tex[2], pos[2], pt[2];
    for (int i = 0; i < 2; i++) {
        pl_tex tx = frames[i]->tex;
        enum pl_tex_sample_mode mode = PL_TEX_SAMPLE_NEAREST;
        if (tx->params.format->caps & PL_FMT_CAP_LINEAR)
            mode = PL_TEX_SAMPLE_LINEAR;
        tex[i] = sh_bind(sh, tx, PL_TEX_ADDRESS_CLAMP, mode, "frame", NULL,
                         &pos[i], &pt[i]);
        if (!tex[i])
            return false;
    }

    ident_t gm = NULL_IDENT;
    if (lm->global > 0.0f)
        gm = sh_motion_global(sh, *input->state, PL_MOTION_FORWARD);

    // The flow fields need not measure the same as the frames.
    const int flow_w = flows[0]->params.w, flow_h = flows[0]->params.h;
    const float flow_sx = (float) flow_w / tpars->w;
    const float flow_sy = (float) flow_h / tpars->h;
    // Distances are given in pixels of the flow fields, so that the result
    // does not depend on the scale the motion was estimated at.
    const float pitch = PL_MAX(lm->pitch, 1.0f);
    const float pitch_x = pitch / flow_sx, pitch_y = pitch / flow_sy;
    const int iters = PL_MAX(lm->iterations, 0);
    const bool photo = lm->sigma > 0.0f;
    const bool sharp = lm->sharp > 0.0f;
    const float inv_sigma2 = 1.0f / PL_MAX(PL_SQUARE(lm->sigma), 1e-8f);
    const float inv_sharp2 = flow_sx * flow_sy /
                             PL_MAX(PL_SQUARE(lm->sharp), 1e-8f);
    const float prior_max = 1.0f + (gm ? lm->global : 0.0f);
    const int num = 4 * ndir + (gm ? 1 : 0);
    // EXPERIMENT: interpretation of the flow's z/w channels, see REPORT.md
    const int repr = getenv("LM_REPR") ? atoi(getenv("LM_REPR")) : 0;

    // clang-format off
#pragma GLSLH /* frame_interp_local */                                          \
    vec4 lm_sample_a(vec2 p) { return ${conv[0]}(textureLod(${tex[0]}, p, 0.0)); }\
    vec4 lm_sample_b(vec2 p) { return ${conv[1]}(textureLod(${tex[1]}, p, 0.0)); }\
                                                                                \
    float lm_inside(vec2 p) {                                                   \
        return all(greaterThanEqual(p, vec2(0.0))) &&                           \
               all(lessThanEqual(p, vec2(1.0))) ? 1.0 : 0.0;                    \
    }                                                                           \
                                                                                \
    float lm_luma(vec4 c) {                                                     \
        return sqrt(max(dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0));        \
    }                                                                           \
                                                                                \
    /* Mix both frames compensated by the motion `mv` (in source pixels), and   \
     * return the mismatch between the two samples in `err`. */                 \
    vec4 lm_pair(vec2 pa, vec2 pb, vec2 mv, float t, vec2 vis, out float err) { \
        vec2 qa = pa - t * mv * ${pt[0]};                                       \
        vec2 qb = pb + (1.0 - t) * mv * ${pt[1]};                               \
        vec4 ca = lm_sample_a(qa);                                              \
        vec4 cb = lm_sample_b(qb);                                              \
        float wa = (1.0 - t) * lm_inside(qa) * vis.x;                           \
        float wb = t * lm_inside(qb) * vis.y;                                   \
        /* Without both samples there is no evidence either way. */             \
        err = wa > 0.0 && wb > 0.0                                              \
            ? mix(${float: lm->sigma}, abs(lm_luma(ca) - lm_luma(cb)),          \
                  min(vis.x, vis.y))                                            \
            : ${float: lm->sigma};                                              \
        return wa + wb > 0.0 ? (wa * ca + wb * cb) / (wa + wb)                  \
                             : mix(ca, cb, t);                                  \
    }
    // clang-format on

    ident_t cubic_id[2] = {0};
    if (sharp) {
        ident_t *cubic = cubic_id;
        for (int i = 0; i < 2; i++) {
            cubic[i] = sh_fresh(sh, "lm_cubic");
            // clang-format off
#pragma GLSLH /* frame_interp_local */                                          \
    /* Catmull-Rom, with the inner 2x2 taps folded into one bilinear fetch. */  \
    vec4 ${cubic[i]}(vec2 p) {                                                  \
        vec2 pt = ${pt[i]};                                                     \
        vec2 c  = floor(p / pt - 0.5) + 0.5;                                    \
        vec2 f  = p / pt - c;                                                   \
        vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));                             \
        vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);                               \
        vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));                              \
        vec2 w3 = f * f * (-0.5 + 0.5 * f);                                     \
        vec2 w12 = w1 + w2;                                                     \
        vec2 p0  = (c - 1.0) * pt;                                              \
        vec2 p12 = (c + w2 / w12) * pt;                                         \
        vec2 p3  = (c + 2.0) * pt;                                              \
        return ${conv[i]}(                                                      \
            (textureLod(${tex[i]}, vec2(p0.x,  p0.y),  0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p0.y),  0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p0.y),  0.0) * w3.x) * w0.y +    \
            (textureLod(${tex[i]}, vec2(p0.x,  p12.y), 0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p12.y), 0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p12.y), 0.0) * w3.x) * w12.y +   \
            (textureLod(${tex[i]}, vec2(p0.x,  p3.y),  0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p3.y),  0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p3.y),  0.0) * w3.x) * w3.y);    \
    }
            // clang-format on
        }

        // clang-format off
#pragma GLSLH /* frame_interp_local */                                          \
    vec4 lm_pair_sharp(vec2 pa, vec2 pb, vec2 mv, float t, vec2 vis) {          \
        vec2 qa = pa - t * mv * ${pt[0]};                                       \
        vec2 qb = pb + (1.0 - t) * mv * ${pt[1]};                               \
        vec4 ca = ${cubic[0]}(qa);                                              \
        vec4 cb = ${cubic[1]}(qb);                                              \
        float wa = (1.0 - t) * lm_inside(qa) * vis.x;                           \
        float wb = t * lm_inside(qb) * vis.y;                                   \
        return wa + wb > 0.0 ? (wa * ca + wb * cb) / (wa + wb)                  \
                             : mix(ca, cb, t);                                  \
    }
        // clang-format on
    }

    // clang-format off
#pragma GLSL /* frame_interp_local */                                           \
    vec4 color;                                                                 \
    {                                                                           \
    float t = ${dynamic float: t};                                              \
    vec2 pa = ${pos[0]}, pb = ${pos[1]};                                        \
    /* Candidates: their sample, motion, weight and mismatch. */                \
    vec4 ck[9];                                                                 \
    vec2 vk[9];                                                                 \
    float wk[9], ek[9];                                                         \
    vec2 sk[9];                                                                 \
    ivec2 last = ivec2(${const int: flow_w}, ${const int: flow_h}) - 1;         \
    vec2 fscale = vec2(${const float: flow_sx}, ${const float: flow_sy});       \
    /* The lattice, in source pixels. */                                        \
    vec2 pitch = vec2(${float: pitch_x}, ${float: pitch_y});                    \
    vec2 grid = pa * vec2(${const float: tpars->w}, ${const float: tpars->h})   \
              / pitch - 0.5;                                                    \
    vec2 base = floor(grid);                                                    \
    vec2 frac = grid - base;
    // clang-format on

    for (int d = 0; d < ndir; d++) {
        ident_t flow = bind_tex(sh, "lm_flow", flows[d]);
        // The backward field holds the displacement from frame `b` to `a`,
        // and its content has `1 - t` of the frame interval to go.
        const float sign = d ? -1.0f : 1.0f;
        // clang-format off
#pragma GLSL /* frame_interp_local */                                           \
    for (int i = 0; i < 4; i++) {                                               \
        vec2 off = vec2(i & 1, i >> 1);                                         \
        vec2 x = (base + off + 0.5) * pitch;                                    \
        float tau = abs(${const float: d} - t);                                 \
        vec4 f = texelFetch($flow, clamp(ivec2(floor(x * fscale)),              \
                                         ivec2(0), last), 0);                   \
        for (int n = 0; n < ${const int: iters}; n++) {                         \
            f = texelFetch($flow, clamp(ivec2(floor((x - tau * f.xy) * fscale)),\
                                        ivec2(0), last), 0);                    \
        }                                                                       \
        vec2 w = mix(1.0 - frac, frac, off);                                    \
        int k = ${const int: 4 * d} + i;                                        \
        vk[k] = ${const float: sign} * f.xy;                                    \
        wk[k] = w.x * w.y * ${const float: 1.0f / ndir};                        \
        float vz = clamp(f.z, 0.0, 1.0), vw = 1.0 - clamp(f.w, 0.0, 1.0);       \
        float conf = smoothstep(${float: lm->conf_lo}, ${float: lm->conf_hi},   \
                                vz * vw);                                       \
        float far = 1.0;                                                        \
        if (${const int: repr} == 0 || ${const int: repr} == 4)                 \
            wk[k] *= conf;                                                      \
        if (${const int: repr} == 1 || ${const int: repr} == 4)                 \
            far = vz * vw;                                                      \
        if (${const int: repr} == 2)                                            \
            far = vz;                                                           \
        sk[k] = ${const float: d} < 0.5 ? vec2(1.0, far) : vec2(far, 1.0);       \
        ck[k] = lm_pair(pa, pb, vk[k], t, sk[k], ek[k]);                        \
    }
        // clang-format on
    }

    if (gm) {
        // clang-format off
#pragma GLSL /* frame_interp_local */                                           \
    vk[${const int: num - 1}] = $gm.xy;                                         \
    wk[${const int: num - 1}] = ${float: lm->global};                           \
    sk[${const int: num - 1}] = vec2(1.0);                                      \
    ck[${const int: num - 1}] = lm_pair(pa, pb, $gm.xy, t, sk[${const int: num - 1}], ek[${const int: num - 1}]);
        // clang-format on
    }

    // clang-format off
#pragma GLSL /* frame_interp_local */                                           \
    @if (photo) {                                                               \
        float emin = 1e9;                                                       \
        for (int k = 0; k < ${const int: num}; k++) {                           \
            if (wk[k] > 0.0)                                                    \
                emin = min(emin, ek[k]);                                        \
        }                                                                       \
        for (int k = 0; k < ${const int: num}; k++) {                           \
            wk[k] *= exp((emin * emin - ek[k] * ek[k]) * ${float: inv_sigma2}); \
        }                                                                       \
    @}                                                                          \
    vec4 mc = vec4(0.0);                                                        \
    vec2 mv = vec2(0.0);                                                        \
    vec2 ms = vec2(0.0);                                                        \
    float sum = 0.0;                                                            \
    for (int k = 0; k < ${const int: num}; k++) {                               \
        mc += wk[k] * ck[k];                                                    \
        mv += wk[k] * vk[k];                                                    \
        ms += wk[k] * sk[k];                                                    \
        sum += wk[k];                                                           \
    }                                                                           \
    @if (sharp) {                                                               \
        if (sum > 1e-6) {                                                       \
            mv /= sum;                                                          \
            float var = 0.0;                                                    \
            for (int k = 0; k < ${const int: num}; k++) {                       \
                vec2 dv = vk[k] - mv;                                           \
                var += wk[k] * dot(dv, dv);                                     \
            }                                                                   \
            float agree = exp(-var / sum * ${float: inv_sharp2});               \
            if (agree > 0.004)                                                  \
                mc = mix(mc, sum * lm_pair_sharp(pa, pb, mv, t, ms / sum), agree);\
        }                                                                       \
    @}                                                                          \
    float fb = ${float: lm->fallback}                                           \
             * max(1.0 - sum / ${float: prior_max}, 0.0);                       \
    color = mix(lm_sample_a(pa), lm_sample_b(pb), t);                           \
    if (sum + fb > 1e-6)                                                        \
        color = (mc + fb * color) / (sum + fb);                                 \
    }
    // clang-format on

    return true;
}

// Samples each frame once per pixel, at the position the splat's winner at
// this pixel was displaced from, and mixes the two samples.
// EXPERIMENT (2026-11-13): gather and soft splat variants on stage B's fields
// and the splat's layer marks, see /z/ai/claude/2026-11-13-softsplat.
//
// X_GATHER=1: lattice style gather (as frame_interp_local). Each side of a
// candidate counts as far as the pixel it samples moves with the candidate
// (X_GSV px tolerance, X_GVF floor), and candidates landing on front marked
// pixels get exp(X_GBETA).
// X_SOFT=1: confidence weighted soft splat of every source pixel's colour
// (bilinear footprint, fixed point atomics), holes from the gather.

// Soft splat of both frames to `t` into `splat.acc` (r, g, b, w per pixel).
static bool soft_scatter(struct sh_motion_obj *obj, pl_dispatch dp,
                         const struct pl_frame_interp_input *input, float t)
{
    pl_gpu gpu = input->gpu;
    const int w = obj->flow.dense_fwd->params.w, h = obj->flow.dense_fwd->params.h;
    const size_t n = (size_t) w * h;
    struct pl_buffer_var acc_var = uint_array_var("soft_acc", 4, n);
    struct pl_buffer_var src_var = uint_array_var("splat_src", 2, 2 * n);
    if (!pl_buf_recreate(gpu, &obj->splat.acc, pl_buf_params(
            .size = acc_var.layout.size, .storable = true)))
        return false;

    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "soft splat clear");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    bind_buf(sh, "soft_acc_buf", obj->splat.acc, PL_DESC_ACCESS_WRITEONLY, &acc_var, 1);
#pragma GLSL /* soft clear */                                                   \
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);                                \
    if (all(lessThan(pos, ivec2(${const int: w}, ${const int: h}))))            \
        soft_acc[pos.y * ${const int: w} + pos.x] = uvec4(0u);
    if (!pl_dispatch_compute(dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 })))
        return false;

    const float beta = b_envf("X_SBETA", 3.0f), scost = b_envf("X_SCOST", 16.0f);
    const float shid = b_envf("X_SHID", 0.1f);
    const int two = dc_env("X_STWO", 1), foot = dc_env("X_SFOOT", 1);
    sh = pl_dispatch_begin(dp);
    sh_describe(sh, "soft splat");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    ident_t fa = bind_tex_linear(sh, "soft_a", input->a.tex);
    ident_t fb = bind_tex_linear(sh, "soft_b", input->b.tex);
    bind_buf(sh, "splat_src_buf", obj->splat.src, PL_DESC_ACCESS_READONLY, &src_var, 1);
    bind_buf(sh, "soft_acc_buf", obj->splat.acc, PL_DESC_ACCESS_READWRITE, &acc_var, 1);
#pragma GLSLH /* soft splat */                                                  \
    void soft_add(ivec2 q, vec3 c, float wt) {                                  \
        ivec2 size = ivec2(${const int: w}, ${const int: h});                   \
        if (wt <= 0.0 || any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size)))\
            return;                                                             \
        uint wq = uint(wt * 256.0 + 0.5);                                       \
        if (wq == 0u)                                                           \
            return;                                                             \
        int i = q.y * size.x + q.x;                                             \
        uvec3 cq = uvec3(clamp(c, 0.0, 1.0) * 1023.0 + 0.5) * wq;               \
        atomicAdd(soft_acc[i].x, cq.x);                                         \
        atomicAdd(soft_acc[i].y, cq.y);                                         \
        atomicAdd(soft_acc[i].z, cq.z);                                         \
        atomicAdd(soft_acc[i].w, wq);                                           \
    }
#pragma GLSL /* soft splat */                                                   \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    float t    = ${float: t};                                                   \
    vec2 isz   = 1.0 / vec2(size);                                              \
    for (int d = 0; d < 2; d++) {                                               \
        if (any(greaterThanEqual(pos, size)))                                   \
            break;                                                              \
        uvec2 src = splat_src[d * size.x * size.y + pos.y * size.x + pos.x];    \
        vec2 f = unpackHalf2x16(src.x);                                         \
        float layer = (src.y >> 30) == 2u ? 1.0 : 0.0;                          \
        bool hid = (src.y & ${const uint: SPLAT_HIDDEN}) != 0u;                 \
        float cost = float(src.y >> 21 & 255u);                                 \
        float tt = d == 0 ? t : 1.0 - t;                                        \
        float wt = (1.0 - tt) * exp(${float: beta} * layer)                     \
                 * (hid ? ${float: shid} : exp(-cost / ${float: scost}));       \
        vec2 sp = (vec2(pos) + 0.5) * isz;                                      \
        vec2 op = (vec2(pos) + 0.5 + f) * isz;                                  \
        vec3 c = d == 0 ? textureLod($fa, sp, 0.0).rgb : textureLod($fb, sp, 0.0).rgb;\
        if (${const int: two} != 0 && !hid) {                                   \
            vec3 o = d == 0 ? textureLod($fb, op, 0.0).rgb : textureLod($fa, op, 0.0).rgb;\
            c = mix(c, o, tt);                                                  \
        }                                                                       \
        vec2 x = vec2(pos) + tt * f;                                            \
        if (${const int: foot} == 1) {                                          \
            ivec2 x0 = ivec2(floor(x));                                         \
            vec2 fr = x - vec2(x0);                                             \
            soft_add(x0,               c, wt * (1.0 - fr.x) * (1.0 - fr.y));    \
            soft_add(x0 + ivec2(1, 0), c, wt * fr.x * (1.0 - fr.y));            \
            soft_add(x0 + ivec2(0, 1), c, wt * (1.0 - fr.x) * fr.y);            \
            soft_add(x0 + ivec2(1, 1), c, wt * fr.x * fr.y);                    \
        } else {                                                                \
            soft_add(ivec2(floor(x + 0.5)), c, wt);                             \
        }                                                                       \
    }
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh, .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));
}

// EXPERIMENT: mean mismatch cost of both fields per QB x QB block (smooth gate
// for the gather's veto)
static bool gather_quality(struct sh_motion_obj *obj, pl_dispatch dp, pl_gpu gpu, int qb)
{
    const int w = obj->flow.dense_fwd->params.w, h = obj->flow.dense_fwd->params.h;
    const int gw = PL_DIV_UP(w, qb), gh = PL_DIV_UP(h, qb);
    struct pl_buffer_var q_var = uint_array_var("g_qual", 1, (size_t) gw * gh);
    struct pl_buffer_var src_var = uint_array_var("splat_src", 2, 2 * (size_t) w * h);
    if (!pl_buf_recreate(gpu, &obj->splat.qual, pl_buf_params(
            .size = q_var.layout.size, .storable = true)))
        return false;
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "gather field quality");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 8, 8, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    bind_buf(sh, "splat_src_buf", obj->splat.src, PL_DESC_ACCESS_READONLY, &src_var, 1);
    bind_buf(sh, "g_qual_buf", obj->splat.qual, PL_DESC_ACCESS_WRITEONLY, &q_var, 1);
    const float hcost = b_envf("X_QH", 64.0f);
#pragma GLSL /* gather quality */                                               \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 b = ivec2(gl_GlobalInvocationID.xy);                                  \
    if (all(lessThan(b, ivec2(${const int: gw}, ${const int: gh})))) {          \
        float acc = 0.0, num = 0.0;                                             \
        for (int d = 0; d < 2; d++)                                             \
        for (int y = 0; y < ${const int: qb}; y++)                              \
        for (int x = 0; x < ${const int: qb}; x++) {                            \
            ivec2 p = b * ${const int: qb} + ivec2(x, y);                       \
            if (any(greaterThanEqual(p, size)))                                 \
                continue;                                                       \
            uint pre = splat_src[d * size.x * size.y + p.y * size.x + p.x].y;   \
            bool hid = (pre & ${const uint: SPLAT_HIDDEN}) != 0u;               \
            acc += hid ? ${float: hcost} : float(pre >> 21 & 255u);             \
            num += 1.0;                                                         \
        }                                                                       \
        g_qual[b.y * ${const int: gw} + b.x] = floatBitsToUint(acc / max(num, 1.0));\
    }
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh, .dispatch_size = { PL_DIV_UP(gw, 8), PL_DIV_UP(gh, 8), 1 }));
}

// EXPERIMENT: patch splat. Every S x S patch of each frame moves rigidly with
// the vector at its centre; each covered output pixel is a fragment that
// samples both frames along that vector, as far as the pixels it samples
// move with the patch (X_PSV px). Fragments add up, weighted, in `splat.acc`.
static bool patch_scatter(struct sh_motion_obj *obj, pl_dispatch dp,
                          const struct pl_frame_interp_input *input, float t, int S)
{
    pl_gpu gpu = input->gpu;
    const int w = obj->flow.dense_fwd->params.w, h = obj->flow.dense_fwd->params.h;
    const size_t n = (size_t) w * h;
    struct pl_buffer_var acc_var = uint_array_var("soft_acc", 4, n);
    struct pl_buffer_var src_var = uint_array_var("splat_src", 2, 2 * n);
    if (!pl_buf_recreate(gpu, &obj->splat.acc, pl_buf_params(
            .size = acc_var.layout.size, .storable = true)))
        return false;
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "soft splat clear");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    bind_buf(sh, "soft_acc_buf", obj->splat.acc, PL_DESC_ACCESS_WRITEONLY, &acc_var, 1);
#pragma GLSL /* soft clear */                                                   \
    ivec2 pos = ivec2(gl_GlobalInvocationID.xy);                                \
    if (all(lessThan(pos, ivec2(${const int: w}, ${const int: h}))))            \
        soft_acc[pos.y * ${const int: w} + pos.x] = uvec4(0u);
    if (!pl_dispatch_compute(dp, pl_dispatch_compute_params(
            .shader = &sh, .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 })))
        return false;

    const int gw = PL_DIV_UP(w, S), gh = PL_DIV_UP(h, S);
    const float sv = b_envf("X_PSV", 1.5f), beta = b_envf("X_PBETA", 3.0f);
    const float sigma = b_envf("X_PSIG", 0.2f);
    sh = pl_dispatch_begin(dp);
    sh_describe(sh, "patch splat");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    ident_t fa = bind_tex_linear(sh, "pt_a", input->a.tex);
    ident_t fb = bind_tex_linear(sh, "pt_b", input->b.tex);
    ident_t fwd = bind_tex(sh, "pt_fwd", obj->flow.dense_fwd);
    ident_t bwd = bind_tex(sh, "pt_bwd", obj->flow.dense_bwd);
    bind_buf(sh, "splat_src_buf", obj->splat.src, PL_DESC_ACCESS_READONLY, &src_var, 1);
    bind_buf(sh, "soft_acc_buf", obj->splat.acc, PL_DESC_ACCESS_READWRITE, &acc_var, 1);
#pragma GLSLH /* patch splat */                                                 \
    void soft_add(ivec2 q, vec3 c, float wt) {                                  \
        ivec2 size = ivec2(${const int: w}, ${const int: h});                   \
        if (wt <= 0.0 || any(lessThan(q, ivec2(0))) || any(greaterThanEqual(q, size)))\
            return;                                                             \
        uint wq = uint(wt * 256.0 + 0.5);                                       \
        if (wq == 0u)                                                           \
            return;                                                             \
        int i = q.y * size.x + q.x;                                             \
        uvec3 cq = uvec3(clamp(c, 0.0, 1.0) * 1023.0 + 0.5) * wq;               \
        atomicAdd(soft_acc[i].x, cq.x);                                         \
        atomicAdd(soft_acc[i].y, cq.y);                                         \
        atomicAdd(soft_acc[i].z, cq.z);                                         \
        atomicAdd(soft_acc[i].w, wq);                                           \
    }
#pragma GLSL /* patch splat */                                                  \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 gid  = ivec2(gl_GlobalInvocationID.xy);                               \
    const int S = ${const int: S};                                              \
    ivec2 blk  = gid / (S + 1), o = gid % (S + 1);                              \
    float t    = ${float: t};                                                   \
    vec2 isz   = 1.0 / vec2(size);                                              \
    float isv  = 1.0 / (${float: sv} * ${float: sv});                           \
    for (int d = 0; d < 2; d++) {                                               \
        if (any(greaterThanEqual(blk, ivec2(${const int: gw}, ${const int: gh}))))\
            break;                                                              \
        ivec2 org = blk * S;                                                    \
        ivec2 cen = min(org + S / 2, size - 1);                                 \
        vec2 f = d == 0 ? texelFetch($fwd, cen, 0).xy : texelFetch($bwd, cen, 0).xy;\
        vec2 u = d == 0 ? f : -f;                                               \
        float tt = d == 0 ? t : -(1.0 - t);                                     \
        ivec2 y = ivec2(floor(vec2(org) + tt * u)) + o;                         \
        vec2 sp = vec2(y) - tt * u;                                             \
        if (any(lessThan(sp, vec2(org) - 0.5)) || any(greaterThanEqual(sp, vec2(org + S) - 0.5)))\
            continue;                                                           \
        vec2 c0 = vec2(y);                                                      \
        vec2 qa = c0 - t * u, qb = c0 + (1.0 - t) * u;                          \
        ivec2 ia = clamp(ivec2(floor(qa + 0.5)), ivec2(0), size - 1);           \
        ivec2 ib = clamp(ivec2(floor(qb + 0.5)), ivec2(0), size - 1);           \
        vec2 da = texelFetch($fwd, ia, 0).xy - u;                               \
        vec2 db = texelFetch($bwd, ib, 0).xy + u;                               \
        float ma = exp(-dot(da, da) * isv), mb = exp(-dot(db, db) * isv);       \
        float own = d == 0 ? ma : mb;                                           \
        float va = d == 0 ? 1.0 : ma, vb = d == 0 ? mb : 1.0;                   \
        vec3 ca = textureLod($fa, (qa + 0.5) * isz, 0.0).rgb;                   \
        vec3 cb = textureLod($fb, (qb + 0.5) * isz, 0.0).rgb;                   \
        float wa = (1.0 - t) * va, wb = t * vb;                                 \
        vec3 c = (wa * ca + wb * cb) / max(wa + wb, 1e-6);                      \
        float la = sqrt(max(dot(ca, vec3(0.2126, 0.7152, 0.0722)), 0.0));       \
        float lb = sqrt(max(dot(cb, vec3(0.2126, 0.7152, 0.0722)), 0.0));       \
        float err = abs(la - lb) * min(va, vb) / ${float: sigma};               \
        ivec2 is = d == 0 ? ia : ib;                                            \
        uint pre = splat_src[d * size.x * size.y + is.y * size.x + is.x].y;     \
        float fr = (pre >> 30) == 2u ? 1.0 : 0.0;                               \
        float wt = (d == 0 ? 1.0 - t : t) * own * exp(-err * err)               \
                 * exp(${float: beta} * fr);                                    \
        soft_add(y, c, wt);                                                     \
    }
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh, .dispatch_size = { PL_DIV_UP(gw * (S + 1), 16), PL_DIV_UP(gh * (S + 1), 16), 1 }));
}

static bool frame_interp_gather(pl_shader sh, struct sh_motion_obj *obj,
                                const struct pl_frame_interp_input *input,
                                const struct pl_color_space *out_color, float t)
{
    const int patch = dc_env("X_PATCH", 0);
    const bool soft = dc_env("X_SOFT", 0) || patch;
    if (patch ? !patch_scatter(obj, input->dispatch, input, t, patch)
              : soft && !soft_scatter(obj, input->dispatch, input, t))
        return false;
    const int qb = dc_env("X_QB", 0);
    if (qb && !gather_quality(obj, input->dispatch, input->gpu, qb))
        return false;
    const struct pl_tex_params *tpars = &input->a.tex->params;
    const struct pl_motion_estimate_frame *frames[2] = { &input->a, &input->b };
    ident_t conv[2];
    for (int i = 0; i < 2; i++) {
        pl_shader sub = pl_dispatch_begin_ex(input->dispatch, true);
        if (sub->prefix == sh->prefix) {
            pl_dispatch_abort(input->dispatch, &sub);
            sub = pl_dispatch_begin_ex(input->dispatch, true);
        }
        sh_require(sub, PL_SHADER_SIG_COLOR, 0, 0);
        struct pl_color_repr repr = frames[i]->repr;
        struct pl_color_space csp = frames[i]->color;
        pl_color_space_infer(&csp);
        if (!pl_color_space_equal(&csp, out_color)) {
            pl_shader_set_alpha(sub, &repr, PL_ALPHA_INDEPENDENT);
            pl_shader_color_map_ex(sub, NULL, pl_color_map_args(csp, *out_color));
        }
        pl_shader_set_alpha(sub, &repr, PL_ALPHA_PREMULTIPLIED);
        conv[i] = sh_subpass(sh, sub);
        pl_dispatch_abort(input->dispatch, &sub);
        if (!conv[i])
            return false;
    }
    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;
    sh_describe(sh, "frame interpolation (gather)");
    ident_t tex[2], pos[2], pt[2], cubic[2];
    for (int i = 0; i < 2; i++) {
        tex[i] = sh_bind(sh, frames[i]->tex, PL_TEX_ADDRESS_CLAMP, PL_TEX_SAMPLE_LINEAR,
                         "frame", NULL, &pos[i], &pt[i]);
        if (!tex[i])
            return false;
        cubic[i] = sh_fresh(sh, "cubic");
#pragma GLSLH /* frame_interp_gather */                                         \
    vec4 ${cubic[i]}(vec2 p) {                                                  \
        vec2 pt = ${pt[i]};                                                     \
        vec2 c  = floor(p / pt - 0.5) + 0.5;                                    \
        vec2 f  = p / pt - c;                                                   \
        vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));                             \
        vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);                               \
        vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));                              \
        vec2 w3 = f * f * (-0.5 + 0.5 * f);                                     \
        vec2 w12 = w1 + w2;                                                     \
        vec2 p0  = (c - 1.0) * pt;                                              \
        vec2 p12 = (c + w2 / w12) * pt;                                         \
        vec2 p3  = (c + 2.0) * pt;                                              \
        float wsum = w12.x * w0.y + (w0.x + w12.x + w3.x) * w12.y + w12.x * w3.y;\
        return ${conv[i]}(                                                      \
            (textureLod(${tex[i]}, vec2(p12.x, p0.y),  0.0) * w12.x * w0.y +    \
            (textureLod(${tex[i]}, vec2(p0.x,  p12.y), 0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p12.y), 0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p12.y), 0.0) * w3.x) * w12.y +   \
             textureLod(${tex[i]}, vec2(p12.x, p3.y),  0.0) * w12.x * w3.y) / wsum);\
    }
    }

    pl_tex fields[2] = { obj->flow.dense_fwd, obj->flow.dense_bwd };
    const int w = fields[0]->params.w, h = fields[0]->params.h;
    const size_t n = (size_t) w * h;
    struct pl_buffer_var src_var = uint_array_var("splat_src", 2, 2 * n);
    bind_buf(sh, "splat_src_buf", obj->splat.src, PL_DESC_ACCESS_READONLY, &src_var, 1);
    if (soft) {
        struct pl_buffer_var acc_var = uint_array_var("soft_acc", 4, n);
        bind_buf(sh, "soft_acc_buf", obj->splat.acc, PL_DESC_ACCESS_READONLY, &acc_var, 1);
    }
    ident_t fwd = bind_tex(sh, "g_fwd", fields[0]);
    ident_t bwd = bind_tex(sh, "g_bwd", fields[1]);
    const int qgw = qb ? PL_DIV_UP(w, qb) : 1, qgh = qb ? PL_DIV_UP(h, qb) : 1;
    if (qb) {
        struct pl_buffer_var q_var = uint_array_var("g_qual", 1, (size_t) qgw * qgh);
        bind_buf(sh, "g_qual_buf", obj->splat.qual, PL_DESC_ACCESS_READONLY, &q_var, 1);
    }
    const float qlo = b_envf("X_QLO", 8.0f), qhi = b_envf("X_QHI", 24.0f);
    const float pitch = b_envf("X_GP", 4.0f);
    const int iters = dc_env("X_GIT", 0);
    const int val = dc_env("X_GVAL", 1);
    const float sv = b_envf("X_GSV", 1.5f), vfl = b_envf("X_GVF", 0.1f);
    const float beta = b_envf("X_GBETA", 3.0f);
    const float sigma = b_envf("X_GSIG", 0.2f), sharp = b_envf("X_GSHARP", 1.0f);
    const float fallback = b_envf("X_GFB", 0.05f);
    const int useconf = dc_env("X_GCONF", 0);
    const float softmin = b_envf("X_SMIN", 0.05f);
    const bool surv = dc_env("X_SURV", 0);
    const bool gv4 = dc_env("X_GV4", 0);
    const float slo = b_envf("X_SLO", 0.3f), shi = b_envf("X_SHI", 0.7f);
#pragma GLSLH /* frame_interp_gather */                                         \
    vec4 g_sa(vec2 p) { return ${conv[0]}(textureLod(${tex[0]}, p, 0.0)); }     \
    vec4 g_sb(vec2 p) { return ${conv[1]}(textureLod(${tex[1]}, p, 0.0)); }     \
    float g_in(vec2 q, vec2 sz) {                                               \
        return all(greaterThanEqual(q, vec2(-0.5))) &&                          \
               all(lessThanEqual(q, sz - 0.5)) ? 1.0 : 0.0;                     \
    }                                                                           \
    float g_luma(vec4 c) {                                                      \
        return sqrt(max(dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0));        \
    }                                                                           \
    float g_front(int d, ivec2 q, ivec2 size) {                                 \
        q = clamp(q, ivec2(0), size - 1);                                       \
        uint pre = splat_src[d * size.x * size.y + q.y * size.x + q.x].y;       \
        return (pre >> 30) == 2u ? 1.0 : 0.0;                                   \
    }
#pragma GLSLH /* frame_interp_gather */                                         \
    /* weighted blend of candidates, with the lattice's photometric weighting  \
     * and resampling along the mean where they agree */                       \
    vec4 g_blend(vec4 ck[8], vec2 vk[8], vec2 sk[8], float wk[8], float ek[8],  \
                 vec2 c0, float t, vec2 fsz, vec2 isz, out float sum) {         \
        float emin = 1e9;                                                       \
        for (int k = 0; k < 8; k++)                                             \
            if (wk[k] > 0.0) emin = min(emin, ek[k]);                           \
        sum = 0.0; vec4 mc = vec4(0.0); vec2 mv = vec2(0.0), ms = vec2(0.0);    \
        for (int k = 0; k < 8; k++) {                                           \
            float w = wk[k] * exp((emin * emin - ek[k] * ek[k]) / (${float: sigma} * ${float: sigma}));\
            wk[k] = w;                                                          \
            sum += w; mc += w * ck[k]; mv += w * vk[k]; ms += w * sk[k];        \
        }                                                                       \
        if (sum <= 1e-6)                                                        \
            return vec4(0.0);                                                   \
        mv /= sum; ms /= sum;                                                   \
        if (${float: sharp} > 0.0) {                                            \
            float var = 0.0;                                                    \
            for (int k = 0; k < 8; k++) {                                       \
                vec2 dv = vk[k] - mv;                                           \
                var += wk[k] * dot(dv, dv);                                     \
            }                                                                   \
            float agree = exp(-var / sum / (${float: sharp} * ${float: sharp}));\
            if (agree > 0.004) {                                                \
                vec2 qa = c0 - t * mv, qb = c0 + (1.0 - t) * mv;                \
                vec4 ca = ${cubic[0]}((qa + 0.5) * isz), cb = ${cubic[1]}((qb + 0.5) * isz);\
                float wa = (1.0 - t) * g_in(qa, fsz) * ms.x, wb = t * g_in(qb, fsz) * ms.y;\
                vec4 cp = wa + wb > 1e-6 ? (wa * ca + wb * cb) / (wa + wb) : mix(ca, cb, t);\
                mc = mix(mc, sum * cp, agree);                                  \
            }                                                                   \
        }                                                                       \
        return mc / sum;                                                        \
    }
#pragma GLSL /* frame_interp_gather */                                          \
    vec4 color;                                                                 \
    {                                                                           \
    float t = ${dynamic float: t};                                              \
    vec2 pa = ${pos[0]}, pb = ${pos[1]};                                        \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    vec2 fsz = vec2(size);                                                      \
    vec2 isz = 1.0 / fsz;                                                       \
    vec2 P = pa * fsz;                                                          \
    vec2 c0 = P - 0.5;                                                          \
    float pitch = ${float: pitch};                                              \
    vec2 grid = P / pitch - 0.5;                                                \
    vec2 base = floor(grid), frac = grid - base;                                \
    vec4 ck[8], dk[8]; vec2 vk[8], sk[8], s0[8]; float wk[8], ek[8], w0[8], e0[8];\
    float tot0 = 0.0, totv = 0.0;                                               \
    float gate = 1.0;                                                           \
    @if (qb) {                                                                  \
        vec2 qg = P / ${const float: (float) qb} - 0.5;                         \
        ivec2 q0 = ivec2(floor(qg)); vec2 qf = qg - vec2(q0);                   \
        ivec2 qs = ivec2(${const int: qgw}, ${const int: qgh}) - 1;             \
        float qv = 0.0;                                                         \
        for (int j = 0; j < 4; j++) {                                           \
            ivec2 o = ivec2(j & 1, j >> 1);                                     \
            ivec2 qq = clamp(q0 + o, ivec2(0), qs);                             \
            vec2 bw = mix(1.0 - qf, qf, vec2(o));                               \
            qv += bw.x * bw.y * uintBitsToFloat(g_qual[qq.y * ${const int: qgw} + qq.x]);\
        }                                                                       \
        gate = 1.0 - smoothstep(${float: qlo}, ${float: qhi}, qv);              \
    @}                                                                          \
    for (int k = 0; k < 8; k++) {                                               \
        int d = k >> 2;                                                         \
        vec2 off = vec2(k & 1, (k >> 1) & 1);                                   \
        vec2 x = (base + off + 0.5) * pitch;                                    \
        float tau = d == 0 ? t : 1.0 - t;                                       \
        ivec2 xi = clamp(ivec2(floor(x)), ivec2(0), size - 1);                  \
        vec4 f = d == 0 ? texelFetch($fwd, xi, 0) : texelFetch($bwd, xi, 0);    \
        for (int it = 0; it < ${const int: iters}; it++) {                      \
            xi = clamp(ivec2(floor(x - tau * f.xy)), ivec2(0), size - 1);       \
            f = d == 0 ? texelFetch($fwd, xi, 0) : texelFetch($bwd, xi, 0);     \
        }                                                                       \
        vec2 v = d == 0 ? f.xy : -f.xy;                                         \
        vec2 bw = mix(1.0 - frac, frac, off);                                   \
        float wt = bw.x * bw.y * 0.5;                                           \
        if (${const int: useconf} != 0)                                         \
            wt *= max(f.z, 0.05);                                               \
        vec2 qa = c0 - t * v, qb = c0 + (1.0 - t) * v;                          \
        float ina = g_in(qa, fsz), inb = g_in(qb, fsz);                         \
        float va = ina, vb = inb;                                               \
        ivec2 ia = clamp(ivec2(floor(qa + 0.5)), ivec2(0), size - 1);           \
        ivec2 ib = clamp(ivec2(floor(qb + 0.5)), ivec2(0), size - 1);           \
        vec4 ca = g_sa((qa + 0.5) * isz), cb = g_sb((qb + 0.5) * isz);          \
        float lerr = abs(g_luma(ca) - g_luma(cb));                              \
        /* unvetoed candidate, as the lattice */                                \
        {                                                                       \
            float wa = (1.0 - t) * ina, wb = t * inb;                           \
            e0[k] = wa > 0.0 && wb > 0.0 ? lerr : ${float: sigma};              \
            dk[k] = wa + wb > 1e-6 ? (wa * ca + wb * cb) / (wa + wb) : mix(ca, cb, t);\
            w0[k] = wt; s0[k] = vec2(ina, inb);                                 \
            tot0 += wt * max(ina, inb);                                         \
        }                                                                       \
        if (${const int: val} != 0) {                                           \
            vec2 da = texelFetch($fwd, ia, 0).xy - v;                           \
            vec2 db = texelFetch($bwd, ib, 0).xy + v;                           \
            @if (gv4) {                                                         \
                ivec2 ja = ivec2(floor(qa)), jb = ivec2(floor(qb));             \
                for (int j = 0; j < 4; j++) {                                   \
                    ivec2 o = ivec2(j & 1, j >> 1);                             \
                    vec2 ea = texelFetch($fwd, clamp(ja + o, ivec2(0), size - 1), 0).xy - v;\
                    vec2 eb = texelFetch($bwd, clamp(jb + o, ivec2(0), size - 1), 0).xy + v;\
                    if (dot(ea, ea) < dot(da, da)) da = ea;                     \
                    if (dot(eb, eb) < dot(db, db)) db = eb;                     \
                }                                                               \
            @}                                                                  \
            float isv = 1.0 / (${float: sv} * ${float: sv});                    \
            float ra = float(splat_src[ia.y * size.x + ia.x].y & 255u) / 255.0; \
            float rb = float(splat_src[size.x * size.y + ib.y * size.x + ib.x].y & 255u) / 255.0;\
            va *= 1.0 - gate * ra * (1.0 - exp(-dot(da, da) * isv));            \
            vb *= 1.0 - gate * rb * (1.0 - exp(-dot(db, db) * isv));            \
            totv += wt * max(va, vb);                                           \
            wt *= ${float: vfl} + (1.0 - ${float: vfl}) * max(va, vb);           \
        }                                                                       \
        float fr = max(g_front(0, ia, size) * va, g_front(1, ib, size) * vb);   \
        wt *= exp(${float: beta} * fr * gate);                                  \
        float wa = (1.0 - t) * va, wb = t * vb;                                 \
        float both = min(va, vb);                                               \
        ek[k] = wa > 0.0 && wb > 0.0 ? mix(${float: sigma}, lerr, both) : ${float: sigma};\
        ck[k] = wa + wb > 1e-6 ? (wa * ca + wb * cb) / (wa + wb) : mix(ca, cb, t);\
        vk[k] = v; wk[k] = wt; sk[k] = vec2(va, vb);                            \
    }                                                                           \
    float sum, sum0;                                                            \
    vec4 cv = g_blend(ck, vk, sk, wk, ek, c0, t, fsz, isz, sum);                \
    float fbw = ${float: fallback} * max(1.0 - sum, 0.0);                       \
    color = mix(g_sa(pa), g_sb(pb), t);                                         \
    if (sum + fbw > 1e-6)                                                       \
        color = (cv * sum + fbw * color) / (sum + fbw);                         \
    @if (surv) {                                                                \
        /* share of the candidates' weight that survives validation */          \
        float sv_ = tot0 > 1e-6 ? totv / tot0 : 0.0;                            \
        float keep = smoothstep(${float: slo}, ${float: shi}, sv_);             \
        if (keep < 1.0) {                                                       \
            vec4 c0c = g_blend(dk, vk, s0, w0, e0, c0, t, fsz, isz, sum0);      \
            if (sum0 > 1e-6)                                                    \
                color = mix(c0c, color, keep);                                  \
        }                                                                       \
    @}                                                                          \
    @if (soft) {                                                                \
        ivec2 px = clamp(ivec2(floor(P)), ivec2(0), size - 1);                  \
        uvec4 acc = soft_acc[px.y * size.x + px.x];                             \
        float aw = float(acc.w) / 256.0;                                        \
        if (aw > 0.0) {                                                         \
            vec4 sc = ${conv[0]}(vec4(vec3(acc.xyz) / (float(acc.w) * 1023.0), 1.0));\
            color = mix(color, sc, smoothstep(0.0, ${float: softmin}, aw));      \
        }                                                                       \
    @}                                                                          \
    }
    return true;
}

static bool frame_interp_splat(pl_shader sh,
                               const struct pl_frame_interp_input *input,
                               const struct pl_color_space *out_color)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(*input->state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    const struct pl_tex_params *tpars = &input->a.tex->params;
    if (!obj || obj->flow.valid != PL_MOTION_BOTH) {
        PL_ERR(sh, "pl_shader_frame_interp: no flow fields available!");
        return false;
    }

    const float t = PL_CLAMP(input->t, 0.0f, 1.0f);
    if (obj->splat.dirty && !splat_prepare(obj, input->gpu, input->dispatch))
        return false;
    if (dc_env("X_GATHER", 0))
        return frame_interp_gather(sh, obj, input, out_color, t);
    if (!splat_scatter(obj, input->dispatch, t))
        return false;

    // Emit the color conversion of each frame as a function, so it can be
    // applied to each sample.
    const struct pl_motion_estimate_frame *frames[2] = { &input->a, &input->b };
    ident_t conv[2];
    for (int i = 0; i < 2; i++) {
        pl_shader sub = pl_dispatch_begin_ex(input->dispatch, true);
        if (sub->prefix == sh->prefix) {
            // The first unique identifier may coincide with `sh`'s, which
            // `sh_subpass` rejects.
            pl_dispatch_abort(input->dispatch, &sub);
            sub = pl_dispatch_begin_ex(input->dispatch, true);
        }
        sh_require(sub, PL_SHADER_SIG_COLOR, 0, 0);
        struct pl_color_repr repr = frames[i]->repr;
        struct pl_color_space csp = frames[i]->color;
        pl_color_space_infer(&csp);
        if (!pl_color_space_equal(&csp, out_color)) {
            pl_shader_set_alpha(sub, &repr, PL_ALPHA_INDEPENDENT);
            pl_shader_color_map_ex(sub, NULL, pl_color_map_args(csp, *out_color));
        }
        pl_shader_set_alpha(sub, &repr, PL_ALPHA_PREMULTIPLIED);
        conv[i] = sh_subpass(sh, sub);
        pl_dispatch_abort(input->dispatch, &sub);
        if (!conv[i])
            return false;
    }

    if (!sh_require(sh, PL_SHADER_SIG_NONE, tpars->w, tpars->h))
        return false;
    sh_describe(sh, "frame interpolation (splat)");

    const bool tap5 = true;
    const int hole = dc_env("X_HOLE", 1);
    const int blur = dc_env("X_BLUR", 2), bstride = dc_env("X_BSTRIDE", 1);
    const unsigned blurt = dc_env("X_BLURT", 128);
    const float blurw = b_envf("X_BLURW", 1.0f), blurself = b_envf("X_BLURSELF", 1.0f);
    const int blurh = dc_env("X_BLURH", 1), blurc = dc_env("X_BLURC", 1);
    const bool holechk = dc_env("X_HOLECHK", 0);
    const float softvis = b_envf("X_SOFTVIS", MOTION_SPLAT_SIMILAR);
    ident_t tex[2], pos[2], pt[2], cubic[2];
    for (int i = 0; i < 2; i++) {
        pl_tex tx = frames[i]->tex;
        enum pl_tex_sample_mode mode = PL_TEX_SAMPLE_NEAREST;
        if (tx->params.format->caps & PL_FMT_CAP_LINEAR)
            mode = PL_TEX_SAMPLE_LINEAR;
        tex[i] = sh_bind(sh, tx, PL_TEX_ADDRESS_CLAMP, mode, "frame", NULL,
                         &pos[i], &pt[i]);
        if (!tex[i])
            return false;

        cubic[i] = sh_fresh(sh, "cubic");
        // clang-format off
#pragma GLSLH /* frame_interp_splat */                                          \
    /* Catmull-Rom, with the inner 2x2 taps folded into one bilinear fetch. */  \
    vec4 ${cubic[i]}(vec2 p) {                                                  \
        vec2 pt = ${pt[i]};                                                     \
        vec2 c  = floor(p / pt - 0.5) + 0.5;                                    \
        vec2 f  = p / pt - c;                                                   \
        vec2 w0 = f * (-0.5 + f * (1.0 - 0.5 * f));                             \
        vec2 w1 = 1.0 + f * f * (-2.5 + 1.5 * f);                               \
        vec2 w2 = f * (0.5 + f * (2.0 - 1.5 * f));                              \
        vec2 w3 = f * f * (-0.5 + 0.5 * f);                                     \
        vec2 w12 = w1 + w2;                                                     \
        vec2 p0  = (c - 1.0) * pt;                                              \
        vec2 p12 = (c + w2 / w12) * pt;                                         \
        vec2 p3  = (c + 2.0) * pt;                                              \
        @if (tap5) {                                                            \
        float wsum = w12.x * w0.y + (w0.x + w12.x + w3.x) * w12.y + w12.x * w3.y;\
        return ${conv[i]}(                                                      \
            (textureLod(${tex[i]}, vec2(p12.x, p0.y),  0.0) * w12.x * w0.y +    \
            (textureLod(${tex[i]}, vec2(p0.x,  p12.y), 0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p12.y), 0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p12.y), 0.0) * w3.x) * w12.y +   \
             textureLod(${tex[i]}, vec2(p12.x, p3.y),  0.0) * w12.x * w3.y) / wsum);\
        @} else {                                                               \
        return ${conv[i]}(                                                      \
            (textureLod(${tex[i]}, vec2(p0.x,  p0.y),  0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p0.y),  0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p0.y),  0.0) * w3.x) * w0.y +    \
            (textureLod(${tex[i]}, vec2(p0.x,  p12.y), 0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p12.y), 0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p12.y), 0.0) * w3.x) * w12.y +   \
            (textureLod(${tex[i]}, vec2(p0.x,  p3.y),  0.0) * w0.x  +           \
             textureLod(${tex[i]}, vec2(p12.x, p3.y),  0.0) * w12.x +           \
             textureLod(${tex[i]}, vec2(p3.x,  p3.y),  0.0) * w3.x) * w3.y);    \
        @}                                                                      \
    }
        // clang-format on
    }

    const int w = obj->flow.dense_fwd->params.w, h = obj->flow.dense_fwd->params.h;
    const size_t n = (size_t) w * h;
    struct pl_buffer_var src_var = uint_array_var("splat_src", 2, 2 * n);
    struct pl_buffer_var key_var = uint_array_var("splat_key", 1, n);
    bind_buf(sh, "splat_src_buf", obj->splat.src, PL_DESC_ACCESS_READONLY, &src_var, 1);
    bind_buf(sh, "splat_key_buf", obj->splat.key[obj->splat.frame & 1],
             PL_DESC_ACCESS_READONLY, &key_var, 1);

    // clang-format off
#pragma GLSLH /* frame_interp_splat */                                          \
    float splat_inside(vec2 p) {                                                \
        return all(greaterThanEqual(p, vec2(0.0))) &&                           \
               all(lessThanEqual(p, vec2(1.0))) ? 1.0 : 0.0;                    \
    }                                                                           \
                                                                                \
    float splat_luma(vec4 c) {                                                  \
        return sqrt(max(dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0));        \
    }

#pragma GLSL /* frame_interp_splat */                                           \
    vec4 color;                                                                 \
    {                                                                           \
    float t = ${dynamic float: t};                                              \
    vec2 pa = ${pos[0]}, pb = ${pos[1]};                                        \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 px = clamp(ivec2(floor(pa * vec2(size))), ivec2(0), size - 1);        \
    ivec2 from = px;                                                            \
    uint key = splat_key[px.y * size.x + px.x];                                 \
    if (key == ${const uint: SPLAT_EMPTY}) {                                    \
        /* Nothing landed here: borrow from the neighbour furthest back, */     \
        /* which is what gets uncovered between diverging layers */             \
        uint best = 0u;                                                         \
        for (int r = 1; r <= ${const int: hole} && best == 0u; r++) {           \
        for (int dy = -r; dy <= r; dy++) {                                      \
        for (int dx = -r; dx <= r; dx++) {                                      \
            if (max(abs(dx), abs(dy)) != r)                                     \
                continue;                                                       \
            ivec2 np = clamp(px + ivec2(dx, dy), ivec2(0), size - 1);           \
            uint nk = splat_key[np.y * size.x + np.x];                          \
            uint score = nk == ${const uint: SPLAT_EMPTY} ? 0u : nk ^ 0x3FFFFFFFu;\
            if (score > best) {                                                 \
                best = score;                                                   \
                from = np;                                                      \
            }                                                                   \
        }                                                                       \
        }                                                                       \
        }                                                                       \
        if (best != 0u)                                                         \
            key = best ^ 0x3FFFFFFFu;                                           \
    }                                                                           \
                                                                                \
    bool done = false;                                                          \
    if (key != ${const uint: SPLAT_EMPTY}) {                                    \
        int d = int(key >> 20 & 1u);                                            \
        ivec2 sp = from + ivec2(key & 1023u, key >> 10 & 1023u) - 512;          \
        vec2 f = unpackHalf2x16(splat_src[(d * size.y + sp.y) * size.x + sp.x].x);\
        if (d == 1)                                                             \
            f = -f;                                                             \
        /* Visibility in a and b; the source frame is always visible */         \
        float other = (key & ${const uint: SPLAT_HIDDEN}) != 0u ? 0.0 : 1.0;    \
        vec2 vis = d == 0 ? vec2(1.0, other) : vec2(other, 1.0);                \
        vec2 to_a = -t * f, to_b = (1.0 - t) * f;                               \
        @if (holechk) {                                                         \
        if (from != px) {                                                       \
            /* borrowed vector: per side, look for a pixel moving with it at */ \
            /* or next to its endpoint, and sample there */                     \
            ivec2 ia = ivec2(floor(vec2(px) + to_a + 0.5));                     \
            ivec2 ib = ivec2(floor(vec2(px) + to_b + 0.5));                     \
            vec2 sa = vec2(0.0), sb = vec2(0.0);                                \
            vis = vec2(0.0);                                                    \
            other = 1.0;                                                        \
            for (int i = 0; i < 9; i++) {                                       \
                /* centre first */                                              \
                int j = i == 0 ? 4 : i <= 4 ? i - 1 : i;                        \
                ivec2 o = ivec2(j % 3, j / 3) - 1;                              \
                ivec2 qa2 = clamp(ia + o, ivec2(0), size - 1);                  \
                ivec2 qb2 = clamp(ib + o, ivec2(0), size - 1);                  \
                vec2 fa = unpackHalf2x16(splat_src[qa2.y * size.x + qa2.x].x);  \
                vec2 fb = unpackHalf2x16(splat_src[(size.y + qb2.y) * size.x + qb2.x].x);\
                if (vis.x == 0.0 && length(fa - f) < 1.0) {                     \
                    vis.x = 1.0; sa = vec2(o);                                  \
                }                                                               \
                if (vis.y == 0.0 && length(fb + f) < 1.0) {                     \
                    vis.y = 1.0; sb = vec2(o);                                  \
                }                                                               \
            }                                                                   \
            to_a += sa; to_b += sb;                                             \
        }                                                                       \
        @}                                                                      \
        vec2 qa = pa + to_a * ${pt[0]};                                         \
        vec2 qb = pb + to_b * ${pt[1]};                                         \
        vec2 inside = vec2(splat_inside(qa), splat_inside(qb));                 \
        if (dot(vis, inside) > 0.0) {                                           \
            vec4 ca = ${cubic[0]}(qa), cb = ${cubic[1]}(qb);                    \
            @if (softvis > 0.0) {                                               \
            if (other == 0.0 && inside.x * inside.y > 0.0) {                    \
                /* The hidden side still counts if it shows the same */         \
                float e = (splat_luma(ca) - splat_luma(cb))                     \
                        / ${const float: softvis};                              \
                vis = max(vis, vec2(exp(-e * e)));                              \
            }                                                                   \
            @}                                                                  \
            vec2 wgt = vec2(1.0 - t, t) * vis * inside;                         \
            color = (wgt.x * ca + wgt.y * cb) / (wgt.x + wgt.y);                \
            done = true;                                                        \
            @if (blur > 0) {                                                    \
            /* No trustworthy vector: rather than one arbitrary warp, mix */    \
            /* the warps the pixels around this one would use here */           \
            if (((key & ${const uint: SPLAT_HIDDEN}) == 0u && (key >> 21 & 255u) >= ${const uint: blurt}) ||\
                (${const int: blurc} == 1 && (key & ${const uint: SPLAT_HIDDEN}) != 0u) ||\
                (${const int: blurc} == 2 && (key & ${const uint: SPLAT_HIDDEN}) != 0u &&\
                 (key >> 21 & 255u) != ${const uint: MOTION_SPLAT_COST_HIDDEN}) ||\
                (${const int: blurh} != 0 && from != px)) {                     \
                vec4 sum = color * ${const float: blurself};                    \
                float num = ${const float: blurself};                           \
                for (int i = 0; i < ${const int: (2 * blur + 1) * (2 * blur + 1)}; i++) {\
                    ivec2 o = (ivec2(i % ${const int: 2 * blur + 1}, i / ${const int: 2 * blur + 1}) - ${const int: blur}) * ${const int: bstride};\
                    if (o == ivec2(0))                                          \
                        continue;                                               \
                    ivec2 np = clamp(px + o, ivec2(0), size - 1);               \
                    uint nk = splat_key[np.y * size.x + np.x];                  \
                    if (nk == ${const uint: SPLAT_EMPTY})                       \
                        continue;                                               \
                    int nd = int(nk >> 20 & 1u);                                \
                    ivec2 ns = np + ivec2(nk & 1023u, nk >> 10 & 1023u) - 512;  \
                    vec2 nf = unpackHalf2x16(splat_src[(nd * size.y + ns.y) * size.x + ns.x].x);\
                    if (nd == 1)                                                \
                        nf = -nf;                                               \
                    float no = (nk & ${const uint: SPLAT_HIDDEN}) != 0u ? 0.0 : 1.0;\
                    vec2 nvis = nd == 0 ? vec2(1.0, no) : vec2(no, 1.0);        \
                    vec2 na = pa - t * nf * ${pt[0]};                           \
                    vec2 nb = pb + (1.0 - t) * nf * ${pt[1]};                   \
                    vec2 nw = vec2(1.0 - t, t) * nvis * vec2(splat_inside(na), splat_inside(nb));\
                    if (nw.x + nw.y <= 0.0)                                     \
                        continue;                                               \
                    float rel = (nk & ${const uint: SPLAT_HIDDEN}) == 0u && (nk >> 21 & 255u) < ${const uint: blurt}\
                              ? 1.0 : ${const float: blurw};                    \
                    sum += rel * (nw.x * ${conv[0]}(textureLod(${tex[0]}, na, 0.0)) + \
                            nw.y * ${conv[1]}(textureLod(${tex[1]}, nb, 0.0))) / (nw.x + nw.y);\
                    num += rel;                                                 \
                }                                                               \
                if (num > 0.0)                                                  \
                    color = sum / num;                                          \
            }                                                                   \
            @}                                                                  \
        }                                                                       \
    }                                                                           \
                                                                                \
    /* Nothing to go by: plain mix */                                           \
    if (!done) {                                                                \
        color = mix(${conv[0]}(textureLod(${tex[0]}, pa, 0.0)),                 \
                    ${conv[1]}(textureLod(${tex[1]}, pb, 0.0)), t);             \
    }                                                                           \
    }
    // clang-format on

    return true;
}

bool pl_shader_frame_interp(pl_shader sh,
                            const struct pl_frame_interp_input *input,
                            const struct pl_frame_interp_params *params)
{
    params = PL_DEF(params, &pl_frame_interp_default_params);
    if (!input || !input->state)
        return false;

    enum pl_motion_direction me_dir;
    bool me_global = false;
    switch (params->algo) {
    case PL_FRAME_INTERP_GLOBAL_MOTION:
        me_dir = PL_MOTION_FORWARD;
        me_global = true;
        break;
    case PL_FRAME_INTERP_VISUALIZE:
        me_dir = PL_MOTION_FORWARD;
        break;
    case PL_FRAME_INTERP_LOCAL_MOTION:
        me_dir = params->local.bidir ? PL_MOTION_BOTH : PL_MOTION_FORWARD;
        me_global = params->local.global > 0.0f;
        break;
    case PL_FRAME_INTERP_SPLAT:
        me_dir = PL_MOTION_BOTH;
        break;
    case PL_FRAME_INTERP_ALGO_COUNT:
    default:
        PL_ERR(sh, "pl_shader_frame_interp: invalid algorithm %d!",
               (int) params->algo);
        return false;
    }

    motion_b_request = params->algo == PL_FRAME_INTERP_SPLAT;

    // A no-op if the motion between this pair of frames is already known.
    bool ok = pl_motion_estimate(pl_motion_estimate_input(
        .gpu       = input->gpu,
        .dispatch  = input->dispatch,
        .a         = input->a,
        .b         = input->b,
        .direction = me_dir,
        .estimate_global = me_global,
        .state     = input->state,
    ), &params->motion);
    if (!ok)
        return false;

    struct pl_color_space out_color = input->out_color;
    pl_color_space_infer(&out_color);

    // Each algorithm must fail before modifying `sh` where possible, see
    // `pl_shader_frame_interp` documentation
    switch (params->algo) {
    case PL_FRAME_INTERP_GLOBAL_MOTION:
        return frame_interp_global(sh, input, &out_color);
    case PL_FRAME_INTERP_VISUALIZE:
        return frame_interp_visualize(sh, input, &out_color);
    case PL_FRAME_INTERP_LOCAL_MOTION:
        return frame_interp_local(sh, input, params, &out_color);
    case PL_FRAME_INTERP_SPLAT:
        return frame_interp_splat(sh, input, &out_color);
    case PL_FRAME_INTERP_ALGO_COUNT:
        break;
    }

    pl_unreachable(); // rejected above
}
