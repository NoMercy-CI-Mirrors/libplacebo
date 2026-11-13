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
    // `pl_motion_get_flow` returns. EXPERIMENT: with stage B, they only hold
    // the vector, without the confidence.
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
    pl_tex b_sel[2], b_occ[2], b_sub[2];
    int b_mode;

    // State of `frame_interp_gather`: the layer marks of both fields (pixels in
    // front of others hold `pair`, the pair counter), the fill pass's cost, the
    // marks as read by the interpolation, and the field quality map. See
    // `motion_b_post`. The vectors are in `flow.dense_*` (two components).
    struct {
        pl_buf mark;
        unsigned pair;
        pl_tex cost[2], front[2], quality;
    } gather;

    // State of `frame_interp_mix`, see `motion_b_post_mix`: the quality of
    // each direction's vectors as estimated (`raw`) and as read by the
    // interpolation (`q`, one of `raw` and the scratch textures `tmp`), the
    // filled vectors, and the parameters they were made with.
    struct {
        pl_tex raw[2], tmp[2][2], q[2], vec[2];
        struct pl_frame_interp_mix_params params;
    } mix;

    // Affine model of the forward motion of the whole frame, as an SSBO
    // described by `model_var`, fitted on demand by `motion_model_fit`.
    // `valid` is whether it describes the current frame pair. `acc` is
    // scratch for fitting it.
    struct {
        pl_buf buf;
        pl_buf acc;
        bool valid;
    } model;

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
static pl_fmt flow_fmt_n(pl_gpu gpu, int comps)
{
    static const enum pl_fmt_caps caps = PL_FMT_CAP_STORABLE |
                                         PL_FMT_CAP_LINEAR;

    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_FLOAT, comps, 16, 0, caps);
    if (fmt)
        return fmt;

    return pl_find_fmt(gpu, PL_FMT_FLOAT, comps, 32, 0, caps);
}

static pl_fmt flow_fmt(pl_gpu gpu)
{
    return flow_fmt_n(gpu, 4);
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
    }
    pl_buf_destroy(gpu, &obj->gather.mark);
    for (int i = 0; i < 2; i++) {
        pl_tex_destroy(gpu, &obj->gather.cost[i]);
        pl_tex_destroy(gpu, &obj->gather.front[i]);
    }
    pl_tex_destroy(gpu, &obj->gather.quality);
    for (int i = 0; i < 2; i++) {
        pl_tex_destroy(gpu, &obj->mix.raw[i]);
        pl_tex_destroy(gpu, &obj->mix.tmp[i][0]);
        pl_tex_destroy(gpu, &obj->mix.tmp[i][1]);
        pl_tex_destroy(gpu, &obj->mix.vec[i]);
    }
    pl_buf_destroy(gpu, &obj->model.buf);
    pl_buf_destroy(gpu, &obj->model.acc);
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
    obj->model.valid = false;
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

bool pl_motion_debug_gather_set(pl_shader_obj state, int dir,
                                const void *flow, const uint8_t *front)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || !obj->gather.front[dir] || !obj->gather.quality)
        return false;
    pl_gpu gpu = state->gpu;
    pl_tex ft = dir ? obj->flow.dense_bwd : obj->flow.dense_fwd;
    pl_tex mt = obj->gather.front[dir], qt = obj->gather.quality;
    if (ft->params.format->texel_size != 4)
        return false;
    const size_t n = (size_t) mt->params.w * mt->params.h;
    const size_t mts = mt->params.format->texel_size;
    uint8_t *tmp = pl_zalloc(NULL, PL_MAX(n * mts, (size_t) qt->params.w * qt->params.h *
                                                 qt->params.format->texel_size));
    for (size_t i = 0; i < n; i++) {
        if (mts == 1)
            tmp[i] = front[i] ? 255 : 0;
        else
            ((uint16_t *) tmp)[i] = front[i] ? 0x3C00 : 0; // half 1.0
    }
    bool ok = pl_tex_upload(gpu, pl_tex_transfer_params(.tex = ft, .ptr = (void *) flow)) &&
              pl_tex_upload(gpu, pl_tex_transfer_params(.tex = mt, .ptr = tmp));
    memset(tmp, 0, (size_t) qt->params.w * qt->params.h * qt->params.format->texel_size);
    ok = ok && pl_tex_upload(gpu, pl_tex_transfer_params(.tex = qt, .ptr = tmp));
    pl_free(tmp);
    return ok;
}

bool pl_motion_debug_mix_set(pl_shader_obj state, int dir, const void *flow,
                             const uint8_t *quality)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || !obj->mix.q[dir])
        return false;
    pl_gpu gpu = state->gpu;
    pl_tex ft = dir ? obj->flow.dense_bwd : obj->flow.dense_fwd;
    pl_tex qt = obj->mix.q[dir];
    if (ft->params.format->texel_size != 4 || qt->params.format->texel_size != 2)
        return false;
    return (!flow || pl_tex_upload(gpu, pl_tex_transfer_params(.tex = ft, .ptr = (void *) flow))) &&
           (!quality || pl_tex_upload(gpu, pl_tex_transfer_params(.tex = qt, .ptr = (void *) quality)));
}

pl_tex pl_motion_debug_mix_get(pl_shader_obj state, int dir, bool raw)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj)
        return NULL;
    return raw ? obj->mix.raw[dir] : obj->mix.q[dir];
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

static ident_t bind_tex_linear(pl_shader sh, const char *name, pl_tex tex)
{
    return sh_desc(sh, (struct pl_shader_desc) {
        .desc = { .name = name, .type = PL_DESC_SAMPLED_TEX },
        .binding = { .object = tex, .sample_mode = PL_TEX_SAMPLE_LINEAR,
                     .address_mode = PL_TEX_ADDRESS_CLAMP },
    });
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

// Replace every block's motion vector with the vector of one of its 8
// neighbours: the one that matches the block's own pixels best (smallest SSD).
// This removes vectors the search got wrong, and unlike a median, it keeps the
// motion of thin objects that only a few blocks see.
static bool motion_smooth_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                               pl_tex ref, pl_tex tgt, pl_tex in, pl_tex out)
{
    const int grid_w = out->params.w, grid_h = out->params.h;
    const int tex_w = ref->params.w, tex_h = ref->params.h;

    // A work group covers `gx` x `gy` blocks, with one thread per row
    const int gx = PL_CLAMP(64 / bs, 1, 8), gy = 4;
    const int bw = gx * bs, bh = gy;
    const size_t shmem = bw * bh * 9 * sizeof(float);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector smoothing (%dx%d blocks)", grid_w, grid_h);
    sh->output = PL_SHADER_SIG_NONE;

    if (!sh_try_compute(sh, bw, bh, false, shmem)) {
        PL_ERR(gpu, "pl_motion_estimate: cannot create smoothing shader");
        goto error;
    }

    ident_t src = bind_tex(sh, "flow_in", in);
    ident_t dst = bind_img(sh, "flow_out", out);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_pt, tgt_id = sh_bind(sh, tgt, PL_TEX_ADDRESS_CLAMP,
                                     PL_TEX_SAMPLE_LINEAR, "tgt", NULL, NULL,
                                     &tgt_pt);
    if (!tgt_id)
        goto error;

    // clang-format off
#pragma GLSLH /* motion_smooth_pass */                                          \
    shared float s_ssd[${const int: bw * bh * 9}];

#pragma GLSL /* motion_smooth_pass */                                           \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    ivec2 size = ivec2(${const int: tex_w}, ${const int: tex_h});               \
    ivec2 tid  = ivec2(gl_LocalInvocationID.xy);                                \
    int row    = tid.x % ${const int: bs};                                      \
    ivec2 blk  = ivec2(gl_WorkGroupID.xy) *                                     \
                 ivec2(${const int: gx}, ${const int: gy}) +                    \
                 ivec2(tid.x / ${const int: bs}, tid.y);                        \
    int base   = (tid.y * ${const int: bw} + tid.x) * 9;                        \
                                                                                \
    vec2 mv[9];                                                                 \
    for (int i = 0; i < 9; i++)                                                 \
        mv[i] = texelFetch($src, clamp(blk + ivec2(i % 3, i / 3) - 1,           \
                                       ivec2(0), grid - 1), 0).xy;              \
                                                                                \
    float ssd[9];                                                               \
    for (int k = 0; k < 9; k++)                                                 \
        ssd[k] = 0.0;                                                           \
    for (int j = 0; j < ${const int: bs}; j++) {                                \
        ivec2 pos = min(blk * ${const int: bs} + ivec2(j, row), size - 1);      \
        float r = texelFetch($ref_id, pos, 0).r;                                \
        vec2 c = vec2(pos) + vec2(0.5);                                         \
        for (int k = 0; k < 9; k++) {                                           \
            if (k == 4)                                                         \
                continue;                                                       \
            float d = r - textureLod($tgt_id, (c + mv[k]) * $tgt_pt, 0.0).r;    \
            ssd[k] += d * d;                                                    \
        }                                                                       \
    }                                                                           \
    for (int k = 0; k < 9; k++)                                                 \
        s_ssd[base + k] = ssd[k];                                               \
    barrier();                                                                  \
                                                                                \
    if (all(lessThan(blk, grid)) && row == 0) {                                 \
        vec2 res = mv[4];                                                       \
        float best_ssd = 1e30;                                                  \
        for (int i = 0; i < 9; i++) {                                           \
            ivec2 p = blk + ivec2(i % 3, i / 3) - 1;                            \
            float sum = 0.0;                                                    \
            for (int j = 0; j < ${const int: bs}; j++)                          \
                sum += s_ssd[base + j * 9 + i];                                 \
            bool inside = all(equal(p, clamp(p, ivec2(0), grid - 1)));          \
            if (inside && i != 4 && sum < best_ssd) {                           \
                res = mv[i];                                                    \
                best_ssd = sum;                                                 \
            }                                                                   \
        }                                                                       \
        imageStore($dst, blk, vec4(res, 1.0, 0.0));                             \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(grid_w, gx), PL_DIV_UP(grid_h, gy), 1 }));

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
// no confidence at all. With `plain`, it is left at 0 instead.
static bool motion_refine_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                               pl_tex ref, pl_tex tgt, pl_tex flow, pl_tex out,
                               bool plain)
{
    const int w = out->params.w, h = out->params.h;
    const int grid_w = flow->params.w, grid_h = flow->params.h;
    const int win = MOTION_REFINE_WINDOW;
    // Each block (or part of a block, for blocks larger than 16) is covered by
    // `dim` x `dim` threads that share its candidates; a work group covers
    // `g` x `g` of them.
    const int dim = PL_MIN(bs, 16), n = PL_DIV_UP(bs, dim);
    const int g = n == 1 ? PL_MAX(16 / dim, 1) : 1;
    // Squared differences over the windows of all pixels of a block
    const int tile = dim + win - 1;
    const int per_thread = PL_DIV_UP(PL_SQUARE(tile), PL_SQUARE(dim));

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector refinement (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;

    const size_t shmem = PL_SQUARE(g * tile) * sizeof(float);
    if (!sh_try_compute(sh, g * dim, g * dim, false, shmem)) {
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
    const float margin = MOTION_REFINE_MARGIN * scale;
    const float flat = MOTION_REFINE_FLAT * scale;

    // clang-format off
#pragma GLSLH /* motion_refine_pass */                                          \
    shared float s_diff[${const int: g * g * tile * tile}];
#pragma GLSL /* motion_refine_pass */                                           \
    ivec2 size  = ivec2(${const int: w}, ${const int: h});                      \
    ivec2 grid  = ivec2(${const int: grid_w}, ${const int: grid_h});            \
    ivec2 local = ivec2(gl_LocalInvocationID.xy) % ${const int: dim};           \
    ivec2 gsub  = ivec2(gl_LocalInvocationID.xy) / ${const int: dim};           \
    int sbase   = (gsub.y * ${const int: g} + gsub.x) * ${const int: tile * tile};\
    int lidx    = local.y * ${const int: dim} + local.x;                        \
    ivec2 wg    = ivec2(gl_WorkGroupID.xy);                                     \
    ivec2 blk   = wg / ${const int: n} * ${const int: g} + gsub;                \
    ivec2 sub   = wg % ${const int: n} * ${const int: dim} + local;             \
    ivec2 pos   = blk * ${const int: bs} + sub;                                 \
                                                                                \
    float r[${const int: per_thread}];                                          \
    vec2 c[${const int: per_thread}];                                           \
    for (int l = 0; l < ${const int: per_thread}; l++) {                        \
        int j = lidx + l * ${const int: dim * dim};     \
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
        int j = lidx + l * ${const int: dim * dim};     \
        if (j < ${const int: tile * tile})                                      \
            s_diff[sbase + j] = r[l];                                                   \
    }                                                                           \
    barrier();                                                                  \
    float centre = s_diff[sbase + (local.y + ${const int: win / 2}) * ${const int: tile}\
                          + local.x + ${const int: win / 2}];                   \
    for (int j = 0; j < ${const int: win * win}; j++) {                         \
        float d = s_diff[sbase + (local.y + j / ${const int: win}) * ${const int: tile} \
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
            int j = lidx + l * ${const int: dim * dim}; \
            vec2 pt = (c[l] + cand[k].xy) * $tgt_pt;                            \
            float d = r[l] - textureLod($tgt_id, pt, 0.0).r;                    \
            if (j < ${const int: tile * tile})                                  \
                s_diff[sbase + j] = d * d;                                              \
        }                                                                       \
        barrier();                                                              \
                                                                                \
        float raw = 0.0;                                                        \
        for (int j = 0; j < ${const int: win * win}; j++) {                     \
            raw += s_diff[sbase + (local.y + j / ${const int: win}) * ${const int: tile}\
                          + local.x + j % ${const int: win}];                   \
        }                                                                       \
        @if (!plain)                                                            \
            mean_ssd += raw / 9.0;                                              \
        float ssd = k == 0 ? raw : raw + ${float: margin};                      \
        if (k == 0 || ssd < best_ssd) {                                         \
            best = cand[k];                                                     \
            best_ssd = ssd;                                                     \
            best_raw = raw;                                                     \
        }                                                                       \
    }                                                                           \
                                                                                \
    float conf = 0.0;                                                           \
    @if (!plain) {                                                              \
    float mismatch = sqrt(best_raw * mean_ssd) / (energy + ${float: flat});     \
    conf = 1.0 / (1.0 + mismatch / ${float: MOTION_REFINE_OCCLUSION});          \
    vec2 to = vec2(pos) + vec2(0.5) + best.xy;                                  \
    if (any(lessThan(to, vec2(0.0))) || any(greaterThan(to, vec2(size))))      \
        conf = 0.0;                                                             \
    @}                                                                          \
                                                                                \
    if (all(lessThan(sub, ivec2(${const int: bs}))) && all(lessThan(pos, size)))\
        imageStore($dst, pos, vec4(best.xy, conf, 0.0));
    // clang-format on

    const struct pl_dispatch_compute_params *params = pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = { PL_DIV_UP(grid_w * n, g), PL_DIV_UP(grid_h * n, g), 1 },
    );

    return pl_dispatch_compute(dispatch, params);

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Set by `pl_shader_frame_interp`: whether the estimation runs stage B's
// occlusion and fill passes after the per pixel selection.
static int motion_b_request;

// EXPERIMENT (stage B): the selection window drags the pixels next to a
// motion boundary along with the side of higher contrast. Where a pixel's
// neighbours move differently, let it take a neighbour's vector if the pixel
// itself matches clearly better that way.
// Stage B turns the block field of the search into a per pixel field, in two
// steps (blocks -> MOTION_SUB x MOTION_SUB sub-blocks -> pixels), then marks
// what is hidden in the other frame, fills in the vectors of hidden pixels,
// and finds which pixels pass in front of others.

// Side length of the sub-blocks of stage B's first step.
#define MOTION_SUB 4

// Rings of far candidates of the sub-block selection, at 2, 4, 8 and 16
// blocks away along both axes. They bring in motion that the smoothing
// removed around thin objects. (The diagonals were not worth their cost.)
#define MOTION_SUB_RINGS 4

// Per sub-block, the best of the block vectors around it (the 3x3 blocks
// around the sub-block and MOTION_SUB_RINGS rings further away), by the SSD
// over the sub-block and a one pixel border. The sub-block's own block
// vector wins ties by a margin of one code value squared.
static bool motion_subsel_pass(pl_gpu gpu, pl_dispatch dispatch, int bs,
                               pl_tex ref, pl_tex tgt, pl_tex blocks, pl_tex out)
{
    const int w = ref->params.w, h = ref->params.h;
    const int gw = out->params.w, gh = out->params.h;
    const int S = MOTION_SUB, WIN = S + 2;
    const int ncand = 9 + 4 * MOTION_SUB_RINGS;

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion sub-block selection (%dx%d)", gw, gh);
    sh->output = PL_SHADER_SIG_NONE;
    // A work group covers 8x8 sub-blocks, whose windows (with the border)
    // are read from shared memory
    const int T = 8 * S + 2, TT = T * T;
    if (!sh_try_compute(sh, 8, 8, false, TT * sizeof(float))) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }
    ident_t blk = bind_tex(sh, "blocks", blocks);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex_linear(sh, "tgt", tgt);
    ident_t dst = bind_img(sh, "sub_out", out);
    // clang-format off
#pragma GLSLH /* motion_subsel_pass */                                          \
    shared float s_ref[${const int: TT}];

#pragma GLSL /* motion_subsel_pass */                                           \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 tile0 = ivec2(gl_WorkGroupID.xy) * ${const int: 8 * S} - 1;           \
    for (int i = int(gl_LocalInvocationIndex); i < ${const int: TT}; i += 64) {\
        ivec2 q = clamp(tile0 + ivec2(i % ${const int: T}, i / ${const int: T}),\
                        ivec2(0), size - 1);                                    \
        s_ref[i] = texelFetch($ref_id, q, 0).r;                                 \
    }                                                                           \
    barrier();                                                                  \
    ivec2 lorg = ivec2(gl_LocalInvocationID.xy) * ${const int: S};              \
    ivec2 grid = ivec2(${const int: blocks->params.w}, ${const int: blocks->params.h});\
    ivec2 sb   = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(sb, ivec2(${const int: gw}, ${const int: gh})))) {         \
        ivec2 org = sb * ${const int: S};                                       \
        ivec2 cen = min(org + ${const int: S / 2}, size - 1);                   \
        vec2 pt = 1.0 / vec2(size);                                             \
        const ivec2 near[9] = ivec2[9](ivec2(0, 0),                             \
            ivec2(-1, -1), ivec2(0, -1), ivec2(1, -1), ivec2(-1, 0),            \
            ivec2(1, 0), ivec2(-1, 1), ivec2(0, 1), ivec2(1, 1));               \
        vec2 best = vec2(0.0), own = vec2(0.0), last = vec2(0.0);               \
        float best_cost = 1e9;                                                  \
        for (int k = 0; k < ${const int: ncand}; k++) {                         \
            ivec2 o = near[k < 9 ? k : 2 + 2 * ((k - 9) & 3) - (((k - 9) & 3) >> 1)];\
            if (k >= 9)                                                         \
                o *= 2 << ((k - 9) >> 2);                                       \
            vec2 v = texelFetch($blk, clamp(cen / ${const int: bs} + o,         \
                                            ivec2(0), grid - 1), 0).xy;         \
            /* most candidates repeat the own, previous or best one */          \
            vec2 d0 = v - own, d1 = v - last, d2 = v - best;                    \
            if (k > 0 && (dot(d0, d0) < 0.01 || dot(d1, d1) < 0.01 ||           \
                          dot(d2, d2) < 0.01))                                  \
                continue;                                                       \
            if (k == 0)                                                         \
                own = v;                                                        \
            last = v;                                                           \
            float cost = 0.0, d;                                                \
            ivec2 q;                                                            \
            int l0 = lorg.y * ${const int: T} + lorg.x;                         \
            @for (y < WIN) {                                                    \
                @for (x < WIN) {                                                \
                    q = clamp(org + ivec2(@x - 1, @y - 1), ivec2(0), size - 1); \
                    d = s_ref[l0 + @y * ${const int: T} + @x]                   \
                      - textureLod($tgt_id, (vec2(q) + 0.5 + v) * pt, 0.0).r;   \
                    cost += d * d * 255.0;                                      \
                @}                                                              \
                /* the cost only grows: stop once this candidate lost */        \
                if (cost >= best_cost * float(${const int: WIN * WIN}))         \
                    continue;                                                   \
            @}                                                                  \
            cost = cost / float(${const int: WIN * WIN})                            \
                 + (k == 0 ? 0.0 : ${const float: 1.0f / 255.0f});              \
            if (cost < best_cost) {                                             \
                best_cost = cost;                                               \
                best = v;                                                       \
            }                                                                   \
        }                                                                       \
        imageStore($dst, sb, vec4(best, 1.0, 0.0));                             \
    }
    // clang-format on
    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh, .dispatch_size = { PL_DIV_UP(gw, 8), PL_DIV_UP(gh, 8), 1 }));
}

// Marks the pixels of `flow` that are hidden in the other frame, in `out`
// (visibility, hidden below 0.5). A pixel is hidden if
// following its vector and then the opposite field's (`opp`, the best of the
// 2x2 pixels around the landing point) does not lead back to it within about
// 1.5 px, unless the content where it does lead back matches its own window
// in `ref` (as on flat areas, where any vector matches).
static bool motion_occlusion_pass(pl_gpu gpu, pl_dispatch dispatch, pl_tex ref,
                                  pl_tex flow, pl_tex opp, pl_tex out)
{
    const int w = out->params.w, h = out->params.h;
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion occlusion (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t opp_id = bind_tex_linear(sh, "flow_opp", opp);
    ident_t ref_id = bind_tex_linear(sh, "ref", ref);
    ident_t dst = bind_img(sh, "flow_out", out);
    // clang-format off
#pragma GLSL /* motion_occlusion_pass */                                        \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(pos, size))) {                                             \
        vec2 pt = 1.0 / vec2(size);                                             \
        vec2 f = texelFetch($src, pos, 0).xy;                                   \
        vec2 rt = f + textureLod($opp_id, (vec2(pos) + 0.5 + f) * pt, 0.0).xy;  \
        ivec2 base = ivec2(floor(vec2(pos) + f));                               \
        for (int i = 0; i < 4; i++) {                                           \
            ivec2 q = clamp(base + ivec2(i & 1, i >> 1), ivec2(0), size - 1);   \
            vec2 r2 = f + texelFetch($opp_id, q, 0).xy;                         \
            if (dot(r2, r2) < dot(rt, rt))                                      \
                rt = r2;                                                        \
        }                                                                       \
        float vis = 1.0 / (1.0 + dot(rt, rt) / 2.25);                           \
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
            float m = ssd / (energy + ${const float: 9.0f / (255.0f * 255.0f)}) / 0.04;\
            vis = max(vis, 1.0 / (1.0 + m * m));                                \
        }                                                                       \
        imageStore($dst, pos, vec4(vis, 0.0, 0.0, 0.0));                        \
    }
    // clang-format on
    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));
}

// Matching cannot find the vector of a pixel hidden in the other frame. Give
// it the vector of the nearest visible pixels around it instead: those found
// in each of 8 directions vote, weighted by their similarity in brightness and
// their distance, for the vector most of them (within 1 px) move with.
//
// Also stores how badly each pixel matches the window its vector leads to in
// `tgt`, relative to the texture of its own window, in `cost` (0-0.25).
static bool motion_fill_pass(pl_gpu gpu, pl_dispatch dispatch, pl_tex ref,
                             pl_tex tgt, pl_tex flow, pl_tex vis, pl_tex out,
                             pl_tex cost)
{
    const int w = out->params.w, h = out->params.h;
    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion vector fill (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dispatch, &sh);
        return false;
    }

    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t vis_id = bind_tex(sh, "vis_in", vis);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex_linear(sh, "tgt", tgt);
    ident_t dst = bind_img(sh, "flow_out", out);
    ident_t cdst = bind_img(sh, "cost_out", cost);
    // clang-format off
#pragma GLSL /* motion_fill_pass */                                             \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    if (all(lessThan(pos, size))) {                                             \
        vec2 f = texelFetch($src, pos, 0).xy;                                   \
        float centre = texelFetch($ref_id, pos, 0).r;                           \
        if (texelFetch($vis_id, pos, 0).x < 0.5) {                              \
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
                    if (texelFetch($vis_id, s, 0).x < 0.5)                      \
                        continue;                                               \
                    float d = (texelFetch($ref_id, s, 0).r - centre)            \
                            / ${const float: MOTION_FILL_SIGMA};                \
                    weight[num] = exp(-d * d) / (1.0 + length(vec2(o)) / 4.0);  \
                    cand[num++] = texelFetch($src, s, 0).xy;                    \
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
                    f = cand[k];                                                \
                }                                                               \
            }                                                                   \
        }                                                                       \
                                                                                \
        float ssd = 0.0, energy = 0.0;                                          \
        for (int j = 0; j < 9; j++) {                                           \
            ivec2 q = clamp(pos + ivec2(j % 3, j / 3) - 1, ivec2(0), size - 1); \
            float r = texelFetch($ref_id, q, 0).r;                              \
            float d = r - textureLod($tgt_id, (vec2(q) + vec2(0.5) + f) / vec2(size), 0.0).r;\
            ssd += d * d;                                                       \
            energy += (r - centre) * (r - centre);                              \
        }                                                                       \
        float c = 0.25 * min(sqrt(ssd / (energy + ${const float: 9.0f / (255.0f * 255.0f)})), 1.0);\
        imageStore($dst, pos, vec4(f, 0.0, 0.0));                               \
        imageStore($cdst, pos, vec4(c, 0.0, 0.0, 0.0));                         \
    }
    // clang-format on
    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));
}

static struct pl_buffer_var uint_array_var(const char *name, int dim_v, size_t n)
{
    return (struct pl_buffer_var) {
        .var = { .name = name, .type = PL_VAR_UINT, .dim_v = dim_v, .dim_m = 1,
                 .dim_a = n },
        .layout = { .offset = 0, .stride = 4 * dim_v, .size = n * 4 * dim_v },
    };
}

// Layer order. A hidden pixel lands on the pixel covering it in the other
// frame, which is thus in front, unless that one's vector leads straight back
// (the two just disagree by a pixel, as on anti-aliased edges). The same goes
// for the pixel the covering one came from in this frame, if that one moves
// along with it. Marks are the pair counter `stamp`, so the buffer never
// needs clearing.
static bool motion_layer_pass(pl_dispatch dp, pl_tex fields[2], pl_tex vis[2],
                              pl_buf mark, unsigned stamp)
{
    const int w = fields[0]->params.w, h = fields[0]->params.h;
    struct pl_buffer_var mark_var = uint_array_var("layer_mark", 1, 2 * (size_t) w * h);
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "motion layer order");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    ident_t fwd = bind_tex(sh, "flow_fwd", fields[0]);
    ident_t bwd = bind_tex(sh, "flow_bwd", fields[1]);
    ident_t vfwd = bind_tex(sh, "vis_fwd", vis[0]);
    ident_t vbwd = bind_tex(sh, "vis_bwd", vis[1]);
    bind_buf(sh, "layer_mark_buf", mark, PL_DESC_ACCESS_WRITEONLY, &mark_var, 1);
    // clang-format off
#pragma GLSL /* motion_layer_pass */                                            \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    int num    = size.x * size.y;                                               \
    for (int d = 0; d < 2; d++) {                                               \
        if (any(greaterThanEqual(pos, size)))                                   \
            break;                                                              \
        vec2 f = d == 0 ? texelFetch($fwd, pos, 0).xy : texelFetch($bwd, pos, 0).xy;\
        float v = d == 0 ? texelFetch($vfwd, pos, 0).x : texelFetch($vbwd, pos, 0).x;\
        ivec2 q = ivec2(floor(vec2(pos) + f + 0.5));                            \
        if (v >= 0.5 || any(lessThan(q, ivec2(0))) ||                           \
            any(greaterThanEqual(q, size)))                                     \
            continue;                                                           \
        vec2 o = d == 0 ? texelFetch($bwd, q, 0).xy : texelFetch($fwd, q, 0).xy;\
        if (length(o + f) <= 0.5)                                               \
            continue;                                                           \
        layer_mark[((1 - d) * size.y + q.y) * size.x + q.x] = ${uint: stamp};   \
        ivec2 r = ivec2(floor(vec2(q) + o + 0.5));                              \
        if (any(lessThan(r, ivec2(0))) || any(greaterThanEqual(r, size)))       \
            continue;                                                           \
        vec2 fr = d == 0 ? texelFetch($fwd, r, 0).xy : texelFetch($bwd, r, 0).xy;\
        if (length(fr + o) < 0.5)                                               \
            layer_mark[(d * size.y + r.y) * size.x + r.x] = ${uint: stamp};     \
    }
    // clang-format on
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));
}

// Block size of the field quality map.
#define MOTION_QUALITY_BLOCK 8

// Turns the layer marks into what the interpolation reads, `front[d]` (1 for
// pixels in front of others, 0 otherwise).
//
// Also writes the field quality map `qual`: the mean cost of both fields
// (`motion_fill_pass`'s mismatch, times 1024, 0-254) per MOTION_QUALITY_BLOCK
// square block, i.e. how well the fields explain the
// frames around there, which decides how far the interpolation trusts them.
static bool motion_pack_pass(pl_dispatch dp, pl_tex cost[2], pl_buf mark,
                             unsigned stamp, pl_tex front[2], pl_tex qual)
{
    const int w = cost[0]->params.w, h = cost[0]->params.h;
    const int qb = MOTION_QUALITY_BLOCK;
    struct pl_buffer_var mark_var = uint_array_var("layer_mark", 1, 2 * (size_t) w * h);
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describe(sh, "motion field packing");
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, qb, qb, false, 2 * qb * qb * sizeof(float))) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }
    ident_t fwd = bind_tex(sh, "cost_fwd", cost[0]);
    ident_t bwd = bind_tex(sh, "cost_bwd", cost[1]);
    ident_t ofwd = bind_img(sh, "front_fwd", front[0]);
    ident_t obwd = bind_img(sh, "front_bwd", front[1]);
    ident_t qdst = bind_img(sh, "quality", qual);
    bind_buf(sh, "layer_mark_buf", mark, PL_DESC_ACCESS_READONLY, &mark_var, 1);
    // clang-format off
#pragma GLSLH /* motion_pack_pass */                                            \
    shared float s_cost[${const int: 2 * qb * qb}];

#pragma GLSL /* motion_pack_pass */                                             \
    ivec2 size = ivec2(${const int: w}, ${const int: h});                       \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy);                               \
    int tid    = int(gl_LocalInvocationIndex);                                  \
    bool inside = all(lessThan(pos, size));                                     \
    for (int d = 0; d < 2; d++) {                                               \
        float cost = 0.0;                                                       \
        if (inside) {                                                           \
            float c = d == 0 ? texelFetch($fwd, pos, 0).x : texelFetch($bwd, pos, 0).x;\
            cost = floor(min(c * 1024.0, 254.0));                               \
            float front = layer_mark[(d * size.y + pos.y) * size.x + pos.x]     \
                          == ${uint: stamp} ? 1.0 : 0.0;                        \
            vec4 res = vec4(front, 0.0, 0.0, 0.0);                              \
            if (d == 0)                                                         \
                imageStore($ofwd, pos, res);                                    \
            else                                                                \
                imageStore($obwd, pos, res);                                    \
        }                                                                       \
        s_cost[d * ${const int: qb * qb} + tid] = cost;                         \
    }                                                                           \
    barrier();                                                                  \
    if (tid == 0) {                                                             \
        ivec2 org = ivec2(gl_WorkGroupID.xy) * ${const int: qb};                \
        ivec2 n = min(size - org, ivec2(${const int: qb}));                     \
        float acc = 0.0;                                                        \
        for (int i = 0; i < ${const int: 2 * qb * qb}; i++)                     \
            acc += s_cost[i];                                                   \
        imageStore($qdst, ivec2(gl_WorkGroupID.xy),                             \
                   vec4(acc / float(2 * n.x * n.y), 0.0, 0.0, 0.0));            \
    }
    // clang-format on
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, qb), PL_DIV_UP(h, qb), 1 }));
}

// Set by `pl_shader_frame_interp` for `motion_b_request` = 2.
static struct pl_frame_interp_mix_params motion_mix_request;

// Whether the parts of `a` and `b` that stage B depends on are the same.
static bool mix_b_params_eq(const struct pl_frame_interp_mix_params *a,
                            const struct pl_frame_interp_mix_params *b)
{
    return a->q_vis == b->q_vis && a->q_cost_lo == b->q_cost_lo &&
           a->q_cost_flat == b->q_cost_flat &&
           a->q_cost_hi == b->q_cost_hi && a->q_outlier == b->q_outlier &&
           a->q_border == b->q_border &&
           a->clean == b->clean &&
           a->soften == b->soften && a->vec_median == b->vec_median &&
           a->fill == b->fill && a->static_thresh == b->static_thresh;
}

// Two component format of the quality fields of `frame_interp_mix`.
static pl_fmt mix_quality_fmt(pl_gpu gpu)
{
    pl_fmt fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 2, 8, 8,
                             PL_FMT_CAP_STORABLE | PL_FMT_CAP_LINEAR);
    return fmt ? fmt : flow_fmt_n(gpu, 2);
}

// Median of 9, per component.
static void sh_median9(pl_shader sh)
{
    // clang-format off
#pragma GLSLH /* sh_median9 */ \
    void m_sort2(inout vec2 a, inout vec2 b) { \
        vec2 lo = min(a, b); \
        b = max(a, b); \
        a = lo; \
    } \
    vec2 m_median9(vec2 p0, vec2 p1, vec2 p2, vec2 p3, vec2 p4, vec2 p5, \
                   vec2 p6, vec2 p7, vec2 p8) { \
        m_sort2(p1, p2); m_sort2(p4, p5); m_sort2(p7, p8); \
        m_sort2(p0, p1); m_sort2(p3, p4); m_sort2(p6, p7); \
        m_sort2(p1, p2); m_sort2(p4, p5); m_sort2(p7, p8); \
        m_sort2(p0, p3); m_sort2(p5, p8); m_sort2(p4, p7); \
        m_sort2(p3, p6); m_sort2(p1, p4); m_sort2(p2, p5); \
        m_sort2(p4, p7); m_sort2(p4, p2); m_sort2(p6, p4); \
        m_sort2(p4, p2); \
        return p4; \
    }
    // clang-format on
}

// Per pixel quality of the vectors of `flow`, into `qual` (x): 1 where the
// vector can be trusted, falling with every sign that it cannot: the pixel is
// hidden in the other frame (`vis`), its window matches badly where the vector
// leads, relative to its own texture, the vector differs from the median of
// those around it, or leaves the frame. See `pl_frame_interp_mix_params`.
//
// `qual` (y) is whether the pixel is static: its surroundings are the same in
// both frames, and not flat. The vectors are copied to `out`, or replaced by their median.
static bool motion_quality_pass(pl_dispatch dp, pl_tex ref, pl_tex tgt,
                                pl_tex flow, pl_tex vis, pl_tex out, pl_tex qual,
                                const struct pl_frame_interp_mix_params *mp)
{
    const int w = out->params.w, h = out->params.h;
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describef(sh, "motion vector quality (%dx%d)", w, h);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    sh_median9(sh);
    ident_t src = bind_tex(sh, "flow_in", flow);
    ident_t vis_id = bind_tex(sh, "vis_in", vis);
    ident_t ref_id = bind_tex(sh, "ref", ref);
    ident_t tgt_id = bind_tex_linear(sh, "tgt", tgt);
    ident_t dst = bind_img(sh, "flow_out", out);
    ident_t qdst = bind_img(sh, "qual_out", qual);
    const bool median = mp->vec_median, outlier = mp->q_outlier > 0.0f;
    const bool cost = mp->q_cost_hi > 0.0f;
    const float inv_outlier = outlier ? 1.0f / PL_SQUARE(mp->q_outlier) : 0.0f;
    const float thresh = PL_MAX(mp->static_thresh, 1e-3f) / 255.0f;
    // clang-format off
#pragma GLSL /* motion_quality_pass */ \
    ivec2 size = ivec2(${const int: w}, ${const int: h}); \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy); \
    if (all(lessThan(pos, size))) { \
        vec2 f = texelFetch($src, pos, 0).xy; \
        float q = 1.0; \
        @if (median || outlier) { \
        ivec2 lo = max(pos - 1, ivec2(0)), hi = min(pos + 1, size - 1); \
        vec2 med = m_median9( \
            texelFetch($src, lo, 0).xy, \
            texelFetch($src, ivec2(pos.x, lo.y), 0).xy, \
            texelFetch($src, ivec2(hi.x, lo.y), 0).xy, \
            texelFetch($src, ivec2(lo.x, pos.y), 0).xy, f, \
            texelFetch($src, ivec2(hi.x, pos.y), 0).xy, \
            texelFetch($src, ivec2(lo.x, hi.y), 0).xy, \
            texelFetch($src, ivec2(pos.x, hi.y), 0).xy, \
            texelFetch($src, hi, 0).xy); \
        vec2 dv = f - med; \
        @if (outlier) \
            q *= exp(-dot(dv, dv) * ${float: inv_outlier}); \
        @if (median) \
            f = med; \
        @} \
        /* nothing to match a vector against once it leaves the frame */ \
        vec2 to = vec2(pos) + vec2(0.5) + f; \
        bool leaves = any(lessThan(to, vec2(0.0))) || any(greaterThan(to, vec2(size))); \
        float qleave = ${float: mp->q_border} * q; \
        if (texelFetch($vis_id, pos, 0).x < 0.5) \
            q *= ${float: 1.0f - PL_CLAMP(mp->q_vis, 0.0f, 1.0f)}; \
        float centre = texelFetch($ref_id, pos, 0).r; \
        float ssd = 0.0, energy = 0.0, sad = 0.0; \
        for (int j = 0; j < 9; j++) { \
            ivec2 p = clamp(pos + ivec2(j % 3, j / 3) - 1, ivec2(0), size - 1); \
            float r = texelFetch($ref_id, p, 0).r; \
            float d = r - textureLod($tgt_id, (vec2(p) + vec2(0.5) + f) / vec2(size), 0.0).r; \
            ssd += d * d; \
            energy += (r - centre) * (r - centre); \
            sad += abs(r - texelFetch($tgt_id, p, 0).r); \
        } \
        @if (cost) { \
        float c = sqrt(ssd / (energy + ${float: 9.0f * PL_SQUARE(PL_MAX(mp->q_cost_flat, 0.1f) / 255.0f)})); \
        q *= 1.0 - smoothstep(${float: mp->q_cost_lo}, ${float: mp->q_cost_hi}, c); \
        @} \
        if (leaves) \
            q = qleave; \
        float still = 1.0 - smoothstep(${float: 0.5f * thresh}, ${float: thresh}, sad / 9.0); \
        /* a flat area is the same in both frames whether it moves or not */ \
        for (int j = 0; j < 4; j++) { \
            ivec2 o = (j & 2) == 0 ? ivec2(3, 0) : ivec2(0, 3); \
            ivec2 p = clamp(pos + ((j & 1) == 0 ? o : -o), ivec2(0), size - 1); \
            float d = texelFetch($ref_id, p, 0).r - centre; \
            energy += d * d; \
        } \
        float flat2 = ${float: 13.0f * PL_SQUARE(PL_MAX(mp->q_cost_flat, 0.1f) / 255.0f)}; \
        still *= energy / (energy + flat2); \
        imageStore($dst, pos, vec4(f, 0.0, 0.0)); \
        imageStore($qdst, pos, vec4(q, still, 0.0, 0.0)); \
    }
    // clang-format on
    return pl_dispatch_compute(dp, pl_dispatch_compute_params(
        .shader = &sh,
        .dispatch_size = { PL_DIV_UP(w, 16), PL_DIV_UP(h, 16), 1 }));
}

enum mix_filter {
    MIX_FILTER_MEDIAN, // 3x3 median of x
    MIX_FILTER_MIN,    // 3x3 minimum of x
    MIX_FILTER_BLUR_H, // gaussian of x and y
    MIX_FILTER_BLUR_V,
};

// Filters a quality field. Isolated pixels of low or high quality are
// estimation noise, which would show as speckle if it decided anything per
// pixel, and the quality should rather be too low than too high around what
// remains.
static bool motion_qfilter_pass(pl_dispatch dp, pl_tex src, pl_tex dst,
                                enum mix_filter filter, float sigma)
{
    const int w = dst->params.w, h = dst->params.h;
    pl_shader sh = pl_dispatch_begin(dp);
    sh_describef(sh, "motion quality filter %d (%dx%d)", (int) filter, w, h);
    sh->output = PL_SHADER_SIG_NONE;
    if (!sh_try_compute(sh, 16, 16, false, 0)) {
        pl_dispatch_abort(dp, &sh);
        return false;
    }

    sh_median9(sh);
    ident_t in = bind_tex(sh, "qual_in", src);
    ident_t out = bind_img(sh, "qual_out", dst);
    const int radius = PL_CLAMP((int) ceilf(2.0f * sigma), 1, 6);
    const bool vert = filter == MIX_FILTER_BLUR_V;
    // clang-format off
#pragma GLSL /* motion_qfilter_pass */ \
    ivec2 size = ivec2(${const int: w}, ${const int: h}); \
    ivec2 pos  = ivec2(gl_GlobalInvocationID.xy); \
    if (all(lessThan(pos, size))) { \
        vec2 res = texelFetch($in, pos, 0).xy; \
        ivec2 lo = max(pos - 1, ivec2(0)), hi = min(pos + 1, size - 1); \
        @if (filter == MIX_FILTER_MEDIAN) { \
        res.x = m_median9( \
            texelFetch($in, lo, 0).xy, \
            texelFetch($in, ivec2(pos.x, lo.y), 0).xy, \
            texelFetch($in, ivec2(hi.x, lo.y), 0).xy, \
            texelFetch($in, ivec2(lo.x, pos.y), 0).xy, res, \
            texelFetch($in, ivec2(hi.x, pos.y), 0).xy, \
            texelFetch($in, ivec2(lo.x, hi.y), 0).xy, \
            texelFetch($in, ivec2(pos.x, hi.y), 0).xy, \
            texelFetch($in, hi, 0).xy).x; \
        @} \
        @if (filter == MIX_FILTER_MIN) { \
        for (int j = 0; j < 9; j++) { \
            ivec2 p = clamp(pos + ivec2(j % 3, j / 3) - 1, ivec2(0), size - 1); \
            res.x = min(res.x, texelFetch($in, p, 0).x); \
        } \
        @} \
        @if (filter >= MIX_FILTER_BLUR_H) { \
        vec2 acc = vec2(0.0); \
        float sum = 0.0; \
        for (int i = -${const int: radius}; i <= ${const int: radius}; i++) { \
            ivec2 p = clamp(pos + i * ivec2(${const int: !vert}, ${const int: vert}), \
                            ivec2(0), size - 1); \
            float g = exp(-float(i * i) * ${float: 0.5f / PL_SQUARE(sigma)}); \
            acc += g * texelFetch($in, p, 0).xy; \
            sum += g; \
        } \
        res = acc / sum; \
        @} \
        imageStore($out, pos, vec4(res, 0.0, 0.0)); \
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

// Stage B after the per pixel field: occlusion, fill, layer order, packing
// and the quality map, all per frame pair.
static bool motion_b_post(pl_gpu gpu, pl_dispatch dispatch,
                          struct sh_motion_obj *obj, pl_tex luma_a, pl_tex luma_b)
{
    pl_tex luma[2] = { luma_a, luma_b };
    pl_tex dense[2] = { obj->flow.dense_fwd, obj->flow.dense_bwd };
    for (int d = 0; d < 2; d++) {
        if (!dense_flow_recreate(gpu, &obj->gather.cost[d], dense[d]->params.w,
                                 dense[d]->params.h, flow_fmt_n(gpu, 1)) ||
            !motion_occlusion_pass(gpu, dispatch, luma[d], obj->b_sel[d],
                                   obj->b_sel[!d], obj->b_occ[d]) ||
            !motion_fill_pass(gpu, dispatch, luma[d], luma[!d], obj->b_sel[d],
                              obj->b_occ[d], dense[d], obj->gather.cost[d]))
            return false;
    }

    const int w = dense[0]->params.w, h = dense[0]->params.h;
    const size_t mark_size = 2 * (size_t) w * h * sizeof(uint32_t);
    if (!obj->gather.mark || obj->gather.mark->params.size != mark_size) {
        // The marks hold the pair counter, so they start out cleared once
        pl_buf_destroy(gpu, &obj->gather.mark);
        void *zero = pl_zalloc(NULL, mark_size);
        obj->gather.mark = pl_buf_create(gpu, pl_buf_params(
            .size = mark_size, .storable = true, .initial_data = zero));
        pl_free(zero);
        obj->gather.pair = 0;
        if (!obj->gather.mark)
            return false;
    }
    const int qb = MOTION_QUALITY_BLOCK;
    pl_fmt front_fmt = pl_find_fmt(gpu, PL_FMT_UNORM, 1, 8, 8, PL_FMT_CAP_STORABLE);
    if (!front_fmt)
        front_fmt = flow_fmt_n(gpu, 1);
    for (int d = 0; d < 2; d++) {
        if (!dense_flow_recreate(gpu, &obj->gather.front[d], w, h, front_fmt))
            return false;
    }
    if (!dense_flow_recreate(gpu, &obj->gather.quality, PL_DIV_UP(w, qb),
                             PL_DIV_UP(h, qb), flow_fmt_n(gpu, 1)))
        return false;
    const unsigned stamp = ++obj->gather.pair;
    if (!motion_layer_pass(dispatch, dense, obj->b_occ, obj->gather.mark, stamp))
        return false;
    return motion_pack_pass(dispatch, obj->gather.cost, obj->gather.mark, stamp,
                            obj->gather.front, obj->gather.quality);
}

// Stage B after the per pixel field, for `frame_interp_mix`: occlusion,
// optionally fill, then the quality of every vector and its cleaning.
static bool motion_b_post_mix(pl_gpu gpu, pl_dispatch dispatch,
                              struct sh_motion_obj *obj, pl_tex luma_a, pl_tex luma_b)
{
    const struct pl_frame_interp_mix_params *mp = &obj->mix.params;
    pl_tex luma[2] = { luma_a, luma_b };
    pl_tex dense[2] = { obj->flow.dense_fwd, obj->flow.dense_bwd };
    pl_fmt qfmt = mix_quality_fmt(gpu);
    if (!qfmt)
        return false;

    for (int d = 0; d < 2; d++) {
        const int w = dense[d]->params.w, h = dense[d]->params.h;
        if (!dense_flow_recreate(gpu, &obj->mix.raw[d], w, h, qfmt) ||
            !dense_flow_recreate(gpu, &obj->mix.tmp[d][0], w, h, qfmt) ||
            !dense_flow_recreate(gpu, &obj->mix.tmp[d][1], w, h, qfmt) ||
            !motion_occlusion_pass(gpu, dispatch, luma[d], obj->b_sel[d],
                                   obj->b_sel[!d], obj->b_occ[d]))
            return false;

        pl_tex vec = obj->b_sel[d];
        if (mp->fill) {
            if (!dense_flow_recreate(gpu, &obj->mix.vec[d], w, h,
                                     dense[d]->params.format) ||
                !dense_flow_recreate(gpu, &obj->gather.cost[d], w, h,
                                     flow_fmt_n(gpu, 1)) ||
                !motion_fill_pass(gpu, dispatch, luma[d], luma[!d], vec,
                                  obj->b_occ[d], obj->mix.vec[d],
                                  obj->gather.cost[d]))
                return false;
            vec = obj->mix.vec[d];
        }

        if (!motion_quality_pass(dispatch, luma[d], luma[!d], vec, obj->b_occ[d],
                                 dense[d], obj->mix.raw[d], mp))
            return false;

        enum mix_filter filters[4];
        int num = 0;
        if (mp->clean >= 1)
            filters[num++] = MIX_FILTER_MEDIAN;
        if (mp->clean >= 2)
            filters[num++] = MIX_FILTER_MIN;
        if (mp->soften > 0.0f) {
            filters[num++] = MIX_FILTER_BLUR_H;
            filters[num++] = MIX_FILTER_BLUR_V;
        }

        pl_tex cur = obj->mix.raw[d];
        for (int i = 0; i < num; i++) {
            pl_tex next = obj->mix.tmp[d][i & 1];
            if (!motion_qfilter_pass(dispatch, cur, next, filters[i], mp->soften))
                return false;
            cur = next;
        }
        obj->mix.q[d] = cur;
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
                    ? motion_b_request : 0;
    if (bmode != obj->b_mode ||
        (bmode == 2 && !mix_b_params_eq(&obj->mix.params, &motion_mix_request)))
    {
        changed = "parameters changed";
        obj->b_mode = bmode;
        obj->sig_a = obj->sig_b = 0;
    }
    obj->mix.params = motion_mix_request;

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
    pl_fmt blk_fmt = flow_fmt_n(gpu, 2);
    if (!mv_fmt || !blk_fmt) {
        PL_ERR(gpu, "pl_motion_estimate: no storable, linearly filterable "
                    "two component float format available!");
        goto error;
    }

    // These are no-ops as long as the motion search shape stays the same.
    // Otherwise, they invalidate the no longer usable temporal prior that resides
    // in the `obj->flow.fwd` and `obj->flow.bwd`.
    if (input->direction & PL_MOTION_FORWARD) {
        if (!flow_pyramid_recreate(gpu, obj, &obj->flow.fwd, w, h, levels,
                                   params->block_size, blk_fmt) ||
            !dense_flow_recreate(gpu, &obj->flow.dense_fwd, w, h,
                                    bmode ? blk_fmt : mv_fmt))
            goto error;
    }

    if (input->direction & PL_MOTION_BACKWARD) {
        if (!flow_pyramid_recreate(gpu, obj, &obj->flow.bwd, w, h, levels,
                                   params->block_size, blk_fmt) ||
            !dense_flow_recreate(gpu, &obj->flow.dense_bwd, w, h,
                                    bmode ? blk_fmt : mv_fmt))
            goto error;
    }

    for (int i = 0; i < 2 && bmode; i++) {
        if (!dense_flow_recreate(gpu, &obj->b_sel[i], w, h, blk_fmt) ||
            !dense_flow_recreate(gpu, &obj->b_occ[i], w, h, flow_fmt_n(gpu, 1)))
            goto error;
    }

    if (!flow_pyramid_recreate(gpu, obj, &obj->scratch.raw, w, h, levels,
                               params->block_size, blk_fmt))
        goto error;

    if (!flow_pyramid_recreate(gpu, obj, &obj->scratch.filtered, w, h, levels,
                               params->block_size, blk_fmt))
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
            pl_tex prior = has_prior ? field->elem[i] : NULL;

            struct motion_geometry geom;
            ok = motion_plan(gpu, tmp, params->block_size, radius,
                             motion_seeds_plan(parent != NULL, prior != NULL),
                             i == 0, out->params.w, out->params.h, &geom);
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

            ok = motion_smooth_pass(gpu, input->dispatch, params->block_size,
                                    fwd ? a : b, fwd ? b : a, raw, out);
            if (!ok)
                break;
        }

        if (!ok)
            break;

        pl_tex prev = obj->scratch.filtered.elem[0];
        pl_tex ra = fwd ? obj->pyramid_prev.elem[0] : obj->scratch.levels.elem[0];
        pl_tex rb = fwd ? obj->scratch.levels.elem[0] : obj->pyramid_prev.elem[0];
        if (bmode) {
            const int S = MOTION_SUB;
            ok = ok && dense_flow_recreate(gpu, &obj->b_sub[d], PL_DIV_UP(w, S),
                                           PL_DIV_UP(h, S), blk_fmt) &&
                 motion_subsel_pass(gpu, input->dispatch, params->block_size,
                                    ra, rb, prev, obj->b_sub[d]) &&
                 motion_refine_pass(gpu, input->dispatch, S, ra, rb,
                                    obj->b_sub[d], obj->b_sel[d], true);
        } else {
            ok = ok && motion_refine_pass(gpu, input->dispatch, params->block_size,
                                          ra, rb, prev,
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

    if (bmode == 1 && !motion_b_post(gpu, input->dispatch, obj, obj->pyramid_prev.elem[0],
                                     obj->scratch.levels.elem[0]))
        goto error;
    if (bmode == 2 && !motion_b_post_mix(gpu, input->dispatch, obj, obj->pyramid_prev.elem[0],
                                         obj->scratch.levels.elem[0]))
        goto error;

    obj->flow.valid = input->direction;
    obj->global.valid = input->estimate_global;
    obj->model.valid = false;

    // Cache frame `b`'s luma pyramid, use reuse `a` buffers as scratch.
    PL_SWAP(obj->pyramid_prev, obj->scratch.levels);
    obj->pyramid_valid = true;

    return true;

error:
    invalidate_temporal(gpu, obj, "estimation failed");
    return false;
}

// The affine model `mv = mat2x3(model) * vec3(pos, 1)` of the global motion,
// with `pos` and `mv` in pixels. `model[0..2]` yields `mv.x` and `model[3..5]`
// yields `mv.y`.
static const struct pl_buffer_var model_var = {
    .var = {
        .name  = "motion_model",
        .type  = PL_VAR_FLOAT,
        .dim_v = 1,
        .dim_m = 1,
        .dim_a = 6,
    },
    .layout = {
        .size   = 6 * sizeof(float),
        .stride = sizeof(float),
    },
};

// Blocks are summed in tiles of `MOTION_MODEL_TILE_DIM` squared, each into
// `MOTION_MODEL_SUMS` floats of `sh_motion_obj.model.acc`.
#define MOTION_MODEL_TILE_DIM 8
#define MOTION_MODEL_SUMS 12

static struct pl_buffer_var model_acc_var(int num_tiles)
{
    return (struct pl_buffer_var) {
        .var = {
            .name  = "model_acc",
            .type  = PL_VAR_FLOAT,
            .dim_v = 1,
            .dim_m = 1,
            .dim_a = num_tiles * MOTION_MODEL_SUMS,
        },
        .layout = {
            .size   = num_tiles * MOTION_MODEL_SUMS * sizeof(float),
            .stride = sizeof(float),
        },
    };
}

// Accumulates the normal equations of a least squares fit of the affine model
// to the blocks of `flow` into `acc`. Unless this is
// the `first` iteration, leaves out the blocks further than `inlier_dist` from
// the model currently in `model`. Positions are scaled to [-1, 1] for the fit,
// to keep the sums well conditioned.
static bool motion_model_acc_pass(pl_gpu gpu, pl_dispatch dispatch, pl_tex flow,
                                  int bs, int w, int h, pl_buf acc,
                                  pl_buf model, bool first, float inlier_dist)
{
    const int grid_w = flow->params.w, grid_h = flow->params.h;
    const int dim = MOTION_MODEL_TILE_DIM;
    const int tiles_w = PL_DIV_UP(grid_w, dim), tiles_h = PL_DIV_UP(grid_h, dim);
    const struct pl_buffer_var acc_var = model_acc_var(tiles_w * tiles_h);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion model accumulation (%dx%d blocks)", grid_w, grid_h);
    sh->output = PL_SHADER_SIG_NONE;

    // One thread per tile: the sums are too few to be worth reducing over
    // shared memory.
    if (!sh_try_compute(sh, 1, 1, false, 0)) {
        PL_ERR(gpu, "pl_shader_frame_interp: cannot create motion model "
               "accumulation shader!");
        goto error;
    }

    ident_t src = bind_tex(sh, "flow_in", flow);
    bind_buf(sh, "model_acc_buf", acc, PL_DESC_ACCESS_WRITEONLY, &acc_var, 1);
    if (!first) {
        bind_buf(sh, "motion_model_buf", model, PL_DESC_ACCESS_READONLY,
                 &model_var, 1);
    }

    // clang-format off
#pragma GLSL /* motion_model_acc_pass */                                        \
    ivec2 grid = ivec2(${const int: grid_w}, ${const int: grid_h});             \
    vec2 size = vec2(${const float: w}, ${const float: h});                     \
    ivec2 tile = ivec2(gl_WorkGroupID.xy);                                      \
    /* Upper triangle of sum(w * p * p^T), then sum(w * p * mv.x) and           \
     * sum(w * p * mv.y), for p = vec3(pos, 1). */                               \
    float s[${const int: MOTION_MODEL_SUMS}];                                   \
    for (int i = 0; i < ${const int: MOTION_MODEL_SUMS}; i++)                   \
        s[i] = 0.0;                                                             \
    for (int y = 0; y < ${const int: dim}; y++) {                               \
        for (int x = 0; x < ${const int: dim}; x++) {                           \
            ivec2 blk = tile * ${const int: dim} + ivec2(x, y);                 \
            if (any(greaterThanEqual(blk, grid)))                               \
                continue;                                                       \
            vec2 flow = texelFetch($src, blk, 0).xy;                            \
            vec2 px = min((vec2(blk) + vec2(0.5)) * ${const float: bs}, size);  \
            vec3 p = vec3(px / size * 2.0 - vec2(1.0), 1.0);                    \
            float w = 1.0;                                                      \
            @if (!first) {                                                      \
                vec3 q = vec3(px, 1.0);                                         \
                vec2 mv = vec2(dot(vec3(motion_model[0], motion_model[1],       \
                                        motion_model[2]), q),                   \
                               dot(vec3(motion_model[3], motion_model[4],       \
                                        motion_model[5]), q));                  \
                if (distance(flow, mv) >= ${float: inlier_dist})                \
                    w = 0.0;                                                    \
            @}                                                                  \
            s[0]  += w * p.x * p.x;                                             \
            s[1]  += w * p.x * p.y;                                             \
            s[2]  += w * p.x;                                                   \
            s[3]  += w * p.y * p.y;                                             \
            s[4]  += w * p.y;                                                   \
            s[5]  += w;                                                         \
            s[6]  += w * p.x * flow.x;                                          \
            s[7]  += w * p.y * flow.x;                                          \
            s[8]  += w * flow.x;                                                \
            s[9]  += w * p.x * flow.y;                                          \
            s[10] += w * p.y * flow.y;                                          \
            s[11] += w * flow.y;                                                \
        }                                                                       \
    }                                                                           \
    int base = (tile.y * ${const int: tiles_w} + tile.x) *                      \
               ${const int: MOTION_MODEL_SUMS};                                 \
    for (int i = 0; i < ${const int: MOTION_MODEL_SUMS}; i++)                   \
        model_acc[base + i] = s[i];
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = { tiles_w, tiles_h, 1 },
    ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Solves the normal equations that `motion_model_acc_pass` left in `acc`, and
// writes the result to `model`, converted to pixel positions.
static bool motion_model_solve_pass(pl_gpu gpu, pl_dispatch dispatch,
                                    pl_buf acc, int num_tiles, int w, int h,
                                    pl_buf model)
{
    const struct pl_buffer_var acc_var = model_acc_var(num_tiles);

    pl_shader sh = pl_dispatch_begin(dispatch);
    sh_describef(sh, "motion model solve (%d tiles)", num_tiles);
    sh->output = PL_SHADER_SIG_NONE;

    // `lanes` threads per sum, each over every `lanes`-th tile
    const int sums = MOTION_MODEL_SUMS, lanes = 16;
    if (!sh_try_compute(sh, lanes, sums, false,
                        sums * lanes * sizeof(float))) {
        PL_ERR(gpu, "pl_shader_frame_interp: cannot create motion model solve "
               "shader!");
        goto error;
    }

    bind_buf(sh, "model_acc_buf", acc, PL_DESC_ACCESS_READONLY, &acc_var, 1);
    bind_buf(sh, "motion_model_buf", model, PL_DESC_ACCESS_WRITEONLY,
             &model_var, 1);

    // clang-format off
#pragma GLSLH /* motion_model_solve_pass */                                     \
    shared float s_sums[${const int: sums}][${const int: lanes}];

#pragma GLSL /* motion_model_solve_pass */                                      \
    int lane = int(gl_LocalInvocationID.x), idx = int(gl_LocalInvocationID.y);  \
    float sum = 0.0;                                                            \
    for (int t = lane; t < ${const int: num_tiles}; t += ${const int: lanes})   \
        sum += model_acc[t * ${const int: sums} + idx];                         \
    s_sums[idx][lane] = sum;                                                    \
    barrier();                                                                  \
                                                                                \
    if (lane == 0 && idx == 0) {                                                \
    float s[${const int: sums}];                                                \
    for (int i = 0; i < ${const int: sums}; i++) {                              \
        s[i] = 0.0;                                                             \
        for (int l = 0; l < ${const int: lanes}; l++)                           \
            s[i] += s_sums[i][l];                                               \
    }                                                                           \
    /* Bias a badly conditioned fit, e.g. to blocks along a single line,        \
     * towards a pure translation. */                                           \
    float reg = 1e-4 * s[5];                                                    \
    mat3 A = mat3(s[0] + reg, s[1],       s[2],                                 \
                  s[1],       s[3] + reg, s[4],                                 \
                  s[2],       s[4],       s[5]);                                \
    /* Without any confident block, there is no motion. */                      \
    vec3 mx = vec3(0.0), my = vec3(0.0);                                        \
    float det = determinant(A);                                                 \
    if (s[5] > 1e-3 && abs(det) > 1e-20) {                                      \
        /* inverse() needs GLSL 140; A is symmetric. */                         \
        mat3 Ai = mat3(A[1][1] * A[2][2] - A[2][1] * A[1][2],                   \
                       A[2][1] * A[0][2] - A[0][1] * A[2][2],                   \
                       A[0][1] * A[1][2] - A[1][1] * A[0][2],                   \
                       A[2][0] * A[1][2] - A[1][0] * A[2][2],                   \
                       A[0][0] * A[2][2] - A[2][0] * A[0][2],                   \
                       A[1][0] * A[0][2] - A[0][0] * A[1][2],                   \
                       A[1][0] * A[2][1] - A[2][0] * A[1][1],                   \
                       A[2][0] * A[0][1] - A[0][0] * A[2][1],                   \
                       A[0][0] * A[1][1] - A[1][0] * A[0][1]) / det;            \
        mx = Ai * vec3(s[6], s[7], s[8]);                                       \
        my = Ai * vec3(s[9], s[10], s[11]);                                     \
    }                                                                           \
    /* From pos * 2 / size - 1 to pixel positions. */                           \
    vec2 scale = vec2(2.0) / vec2(${const float: w}, ${const float: h});        \
    motion_model[0] = mx.x * scale.x;                                           \
    motion_model[1] = mx.y * scale.y;                                           \
    motion_model[2] = mx.z - mx.x - mx.y;                                       \
    motion_model[3] = my.x * scale.x;                                           \
    motion_model[4] = my.y * scale.y;                                           \
    motion_model[5] = my.z - my.x - my.y;                                       \
    }
    // clang-format on

    return pl_dispatch_compute(dispatch, pl_dispatch_compute_params(
        .shader        = &sh,
        .dispatch_size = { 1, 1, 1 },
    ));

error:
    pl_dispatch_abort(dispatch, &sh);
    return false;
}

// Refits of the affine model to the blocks within MOTION_MODEL_INLIER pixels
// of the previous fit.
#define MOTION_MODEL_REFITS 2
#define MOTION_MODEL_INLIER 4.0f

// Fits the affine model of the global motion to the forward block field of
// the current frame pair, into `obj->model.buf`, unless it is already there.
static bool motion_model_fit(pl_gpu gpu, pl_dispatch dispatch,
                             struct sh_motion_obj *obj)
{
    if (obj->model.valid)
        return true;

    pl_tex flow = obj->flow.fwd.elem[0];
    const int num_tiles = PL_DIV_UP(flow->params.w, MOTION_MODEL_TILE_DIM) *
                          PL_DIV_UP(flow->params.h, MOTION_MODEL_TILE_DIM);

    const size_t acc_size = model_acc_var(num_tiles).layout.size;
    if (obj->model.acc && obj->model.acc->params.size != acc_size)
        pl_buf_destroy(gpu, &obj->model.acc);
    if (!obj->model.acc) {
        obj->model.acc = pl_buf_create(gpu, pl_buf_params(
            .size      = acc_size,
            .storable  = true,
            .debug_tag = PL_DEBUG_TAG,
        ));
    }
    if (!obj->model.buf) {
        obj->model.buf = pl_buf_create(gpu, pl_buf_params(
            .size      = model_var.layout.size,
            .storable  = true,
            .debug_tag = PL_DEBUG_TAG,
        ));
    }
    if (!obj->model.acc || !obj->model.buf) {
        PL_ERR(gpu, "pl_shader_frame_interp: failed creating motion model "
               "buffers!");
        return false;
    }

    obj->model.valid = false;
    for (int i = 0; i <= MOTION_MODEL_REFITS; i++) {
        if (!motion_model_acc_pass(gpu, dispatch, flow, obj->params.block_size,
                                   obj->w, obj->h, obj->model.acc,
                                   obj->model.buf, i == 0, MOTION_MODEL_INLIER))
            return false;
        if (!motion_model_solve_pass(gpu, dispatch, obj->model.acc, num_tiles,
                                     obj->w, obj->h, obj->model.buf))
            return false;
    }

    obj->model.valid = true;
    return true;
}

// Mixes the two frames, each sampled at the position offset by the global
// motion, scaled by the frame's distance in time to the intermediate frame.
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
        wk[k] *= conf;                                                          \
        sk[k] = vec2(1.0);                                                      \
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

// Binds both frames of `input` for `frame_interp_gather` and
// `frame_interp_mix`: per frame, the texture, its position and texel size,
// the conversion to `out_color` (`conv`, a function) and a bicubic sampler
// including that conversion (`cubic`).
static bool interp_bind_frames(pl_shader sh,
                               const struct pl_frame_interp_input *input,
                               const struct pl_color_space *out_color,
                               const char *desc, ident_t conv[2], ident_t tex[2],
                               ident_t pos[2], ident_t pt[2], ident_t cubic[2])
{
    const struct pl_tex_params *tpars = &input->a.tex->params;
    const struct pl_motion_estimate_frame *frames[2] = { &input->a, &input->b };
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
    sh_describe(sh, desc);
    for (int i = 0; i < 2; i++) {
        tex[i] = sh_bind(sh, frames[i]->tex, PL_TEX_ADDRESS_CLAMP, PL_TEX_SAMPLE_LINEAR,
                         "frame", NULL, &pos[i], &pt[i]);
        if (!tex[i])
            return false;
        cubic[i] = sh_fresh(sh, "cubic");
        // clang-format off
#pragma GLSLH /* frame_interp_gather */                                         \
    /* Catmull-Rom, the inner 2x2 taps folded into one bilinear fetch and the */\
    /* corners dropped */                                                       \
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
        // clang-format on
    }

    return true;
}

// Every output pixel is a blend of both frames compensated by 8 candidate
// vectors: those of the per pixel fields of both directions at the four points of
// a lattice (pitch MOTION_GATHER_PITCH) around the pixel, as in
// `frame_interp_local`, but read where they are rather than traced back to
// where their content came from.
//
// Each side (frame) of a candidate only counts as far as the pixel it samples
// moves with the candidate. This keeps a candidate from dragging along content
// that moves differently (copies of a glyph over the background moving behind
// it), and makes it fall back to the other frame where its content is hidden.
// Candidates that sample a pixel marked as being in front of others weigh
// more. Both only apply as far as the field quality map trusts the fields, so
// that wrong fields degrade into the blur of plain blending rather than noise.
#define MOTION_GATHER_PITCH 4

// Vector difference, in pixels, at which a side counts for 1/e.
#define MOTION_GATHER_VETO 5.0f

// Weight left to a candidate both of whose sides are vetoed.
#define MOTION_GATHER_FLOOR 0.1f

// Weight of a candidate landing on a front pixel: exp(MOTION_GATHER_FRONT).
#define MOTION_GATHER_FRONT 3.0f

// Range of the quality map's cost over which the fields go from trusted to
// not trusted.
#define MOTION_GATHER_QLO 96.0f
#define MOTION_GATHER_QHI 200.0f

// Tolerated mismatch between the two samples of a candidate, relative to the
// best candidate, in the square root of the luma.
#define MOTION_GATHER_SIGMA 0.2f

// Weight of plain frame mixing, as far as the candidates' weights sum to less
// than 1.
#define MOTION_GATHER_FALLBACK 0.05f

static bool frame_interp_gather(pl_shader sh,
                                const struct pl_frame_interp_input *input,
                                const struct pl_color_space *out_color)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(*input->state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || obj->flow.valid != PL_MOTION_BOTH || !obj->gather.front[0]) {
        PL_ERR(sh, "pl_shader_frame_interp: no flow fields available!");
        return false;
    }
    const float t = PL_CLAMP(input->t, 0.0f, 1.0f);
    ident_t conv[2], tex[2], pos[2], pt[2], cubic[2];
    if (!interp_bind_frames(sh, input, out_color, "frame interpolation (gather)",
                            conv, tex, pos, pt, cubic))
        return false;

    pl_tex *front = obj->gather.front;
    pl_tex qual = obj->gather.quality;
    const int w = front[0]->params.w, h = front[0]->params.h;
    const int qw = qual->params.w, qh = qual->params.h;
    ident_t fwd = bind_tex(sh, "g_fwd", obj->flow.dense_fwd);
    ident_t bwd = bind_tex(sh, "g_bwd", obj->flow.dense_bwd);
    ident_t ffwd = bind_tex(sh, "g_ffwd", front[0]);
    ident_t fbwd = bind_tex(sh, "g_fbwd", front[1]);
    ident_t qmap = bind_tex_linear(sh, "g_qual", qual);
    const float isv = 1.0f / PL_SQUARE(MOTION_GATHER_VETO);
    // clang-format off
#pragma GLSLH /* frame_interp_gather */                                         \
    vec4 g_sa(vec2 p) { return ${conv[0]}(textureLod(${tex[0]}, p, 0.0)); }     \
    vec4 g_sb(vec2 p) { return ${conv[1]}(textureLod(${tex[1]}, p, 0.0)); }     \
    float g_in(vec2 q, vec2 sz) {                                               \
        return all(greaterThanEqual(q, vec2(-0.5))) &&                          \
               all(lessThanEqual(q, sz - 0.5)) ? 1.0 : 0.0;                     \
    }                                                                           \
    float g_luma(vec4 c) {                                                      \
        return sqrt(max(dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0));        \
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
    float gate = 1.0 - smoothstep(${const float: MOTION_GATHER_QLO},            \
        ${const float: MOTION_GATHER_QHI},                                      \
        textureLod($qmap, P / ${const float: MOTION_QUALITY_BLOCK}              \
                        / vec2(${const int: qw}, ${const int: qh}), 0.0).x);    \
    vec2 grid = P / ${const float: MOTION_GATHER_PITCH} - 0.5;                  \
    vec2 base = floor(grid), frac = grid - base;                                \
    vec4 ck[8]; vec2 vk[8], sk[8]; float wk[8], ek[8];                          \
    for (int k = 0; k < 8; k++) {                                               \
        int d = k >> 2;                                                         \
        vec2 off = vec2(k & 1, (k >> 1) & 1);                                   \
        vec2 x = (base + off + 0.5) * ${const float: MOTION_GATHER_PITCH};      \
        ivec2 xi = clamp(ivec2(floor(x)), ivec2(0), size - 1);                  \
        vec2 f = d == 0 ? texelFetch($fwd, xi, 0).xy : texelFetch($bwd, xi, 0).xy;\
        vec2 v = d == 0 ? f : -f;                                               \
        vec2 bw = mix(1.0 - frac, frac, off);                                   \
        float wt = bw.x * bw.y * 0.5;                                           \
        vec2 qa = c0 - t * v, qb = c0 + (1.0 - t) * v;                          \
        float va = g_in(qa, fsz), vb = g_in(qb, fsz);                           \
        ivec2 ia = clamp(ivec2(floor(qa + 0.5)), ivec2(0), size - 1);           \
        ivec2 ib = clamp(ivec2(floor(qb + 0.5)), ivec2(0), size - 1);           \
        vec3 fa = vec3(texelFetch($fwd, ia, 0).xy, texelFetch($ffwd, ia, 0).x); \
        vec3 fb = vec3(texelFetch($bwd, ib, 0).xy, texelFetch($fbwd, ib, 0).x); \
        vec2 da = fa.xy - v, db = fb.xy + v;                                    \
        va *= 1.0 - gate * (1.0 - exp(-dot(da, da) * ${const float: isv}));     \
        vb *= 1.0 - gate * (1.0 - exp(-dot(db, db) * ${const float: isv}));     \
        wt *= ${const float: MOTION_GATHER_FLOOR}                               \
            + ${const float: 1.0f - MOTION_GATHER_FLOOR} * max(va, vb);         \
        wt *= exp(${const float: MOTION_GATHER_FRONT} * gate * max(fa.z * va, fb.z * vb));\
        vec4 ca = g_sa((qa + 0.5) * isz), cb = g_sb((qb + 0.5) * isz);          \
        float wa = (1.0 - t) * va, wb = t * vb;                                 \
        ek[k] = wa > 0.0 && wb > 0.0                                            \
              ? mix(${const float: MOTION_GATHER_SIGMA}, abs(g_luma(ca) - g_luma(cb)), min(va, vb)) \
              : ${const float: MOTION_GATHER_SIGMA};                                                \
        ck[k] = wa + wb > 1e-6 ? (wa * ca + wb * cb) / (wa + wb) : mix(ca, cb, t);\
        vk[k] = v; wk[k] = wt; sk[k] = vec2(va, vb);                            \
    }                                                                           \
    float emin = 1e9;                                                           \
    for (int k = 0; k < 8; k++)                                                 \
        if (wk[k] > 0.0) emin = min(emin, ek[k]);                               \
    float sum = 0.0; vec4 mc = vec4(0.0); vec2 mv = vec2(0.0), ms = vec2(0.0);   \
    for (int k = 0; k < 8; k++) {                                               \
        wk[k] *= exp((emin * emin - ek[k] * ek[k])                              \
                     * ${const float: 1.0f / PL_SQUARE(MOTION_GATHER_SIGMA)});  \
        sum += wk[k]; mc += wk[k] * ck[k]; mv += wk[k] * vk[k]; ms += wk[k] * sk[k];\
    }                                                                           \
    if (sum > 1e-6) {                                                           \
        mv /= sum; ms /= sum;                                                   \
        float var = 0.0;                                                        \
        for (int k = 0; k < 8; k++) {                                           \
            vec2 dv = vk[k] - mv;                                               \
            var += wk[k] * dot(dv, dv);                                         \
        }                                                                       \
        /* where the candidates agree, resample once along their mean with */  \
        /* a sharper filter */                                                  \
        float agree = exp(-var / sum);                                          \
        if (agree > 0.004) {                                                    \
            vec2 qa = c0 - t * mv, qb = c0 + (1.0 - t) * mv;                    \
            vec4 ca = ${cubic[0]}((qa + 0.5) * isz), cb = ${cubic[1]}((qb + 0.5) * isz);\
            float wa = (1.0 - t) * g_in(qa, fsz) * ms.x, wb = t * g_in(qb, fsz) * ms.y;\
            vec4 cp = wa + wb > 1e-6 ? (wa * ca + wb * cb) / (wa + wb) : mix(ca, cb, t);\
            mc = mix(mc, sum * cp, agree);                                      \
        }                                                                       \
    }                                                                           \
    float fbw = ${const float: MOTION_GATHER_FALLBACK} * max(1.0 - sum, 0.0);                       \
    color = mix(g_sa(pa), g_sb(pb), t);                                         \
    if (sum + fbw > 1e-6)                                                       \
        color = (mc + fbw * color) / (sum + fbw);                               \
    }
    // clang-format on
    return true;
}

// out = mix(F, G, Q): G is both frames sampled along the pixel's own vector,
// read from the per pixel fields at the pixel itself, F a fallback blend
// (plain frame mixing unless `mp->fallback`), and Q the quality of the vector
// at the pixels it samples (see `motion_b_post_mix`). With Q = 0 this is F, so
// it can only be worse than F where Q is too high.
//
// The two fields hold the vector of what is at the pixel in their own frame.
// Content that moves by less than its length is at the pixel in at least one
// of the two frames, so one of the two vectors is the right one: the one of
// higher quality (`mp->pick`).
static bool frame_interp_mix(pl_shader sh,
                             const struct pl_frame_interp_input *input,
                             const struct pl_frame_interp_mix_params *mp,
                             const struct pl_color_space *out_color)
{
    struct sh_motion_obj *obj = SH_OBJ_LOOKUP(*input->state, PL_SHADER_OBJ_MOTION,
                                              struct sh_motion_obj);
    if (!obj || obj->flow.valid != PL_MOTION_BOTH || obj->b_mode != 2 ||
        !obj->mix.q[0] || !obj->mix.q[1])
    {
        PL_ERR(sh, "pl_shader_frame_interp: no flow fields available!");
        return false;
    }

    const float t = PL_CLAMP(input->t, 0.0f, 1.0f);
    ident_t conv[2], tex[2], pos[2], pt[2], cubic[2];
    if (!interp_bind_frames(sh, input, out_color, "frame interpolation (mix)",
                            conv, tex, pos, pt, cubic))
        return false;

    const int w = obj->flow.dense_fwd->params.w, h = obj->flow.dense_fwd->params.h;
    ident_t fwd = bind_tex(sh, "m_fwd", obj->flow.dense_fwd);
    ident_t bwd = bind_tex(sh, "m_bwd", obj->flow.dense_bwd);
    ident_t qfwd = bind_tex_linear(sh, "m_qfwd", obj->mix.q[0]);
    ident_t qbwd = bind_tex_linear(sh, "m_qbwd", obj->mix.q[1]);
    const bool consist = mp->consist > 0.0f, veto = mp->veto > 0.0f;
    const float inv_consist = consist ? 1.0f / PL_SQUARE(mp->consist) : 0.0f;
    const float inv_veto = veto ? 1.0f / PL_SQUARE(mp->veto) : 0.0f;
    const float inv_sigma = 1.0f / PL_SQUARE(PL_MAX(mp->sigma, 1e-3f));
    const int pick = mp->pick, fallback = mp->fallback, show = mp->show;
    const int stest = mp->static_test;
    const bool local = fallback == 2 || fallback == 3 || fallback == 5 || fallback == 7;
    const bool sharp = mp->sharp, affine = fallback >= 6;
    const int ncand = fallback == 7 ? 13 : fallback == 5 ? 12 : 10;
    if (affine) {
        bind_buf(sh, "motion_model_buf", obj->model.buf, PL_DESC_ACCESS_READONLY,
                 &model_var, 1);
    }
    const int neigh = PL_CLAMP((int) roundf(mp->neigh), 0, 64);
    ident_t blkf = NULL_IDENT, blkb = NULL_IDENT;
    if (local) {
        blkf = bind_tex_linear(sh, "m_blkf", obj->flow.fwd.elem[0]);
        blkb = bind_tex_linear(sh, "m_blkb", obj->flow.bwd.elem[0]);
    }
    // Size of the block grid in pixels
    const float bgw = obj->grid_w * obj->params.block_size,
                bgh = obj->grid_h * obj->params.block_size;

    // clang-format off
#pragma GLSLH /* frame_interp_mix */ \
    vec4 m_sa(vec2 p) { return ${conv[0]}(textureLod(${tex[0]}, p, 0.0)); } \
    vec4 m_sb(vec2 p) { return ${conv[1]}(textureLod(${tex[1]}, p, 0.0)); } \
    float m_in(vec2 p, vec2 sz) { \
        return all(greaterThanEqual(p, vec2(-0.5))) && \
               all(lessThanEqual(p, sz - 0.5)) ? 1.0 : 0.0; \
    } \
    float m_luma(vec4 c) { \
        return sqrt(max(dot(c.rgb, vec3(0.2126, 0.7152, 0.0722)), 0.0)); \
    } \
    /* Quality of the vector `v` at the pixel centred on `c`, and whether */ \
    /* it samples a static pixel in either frame */ \
    vec2 m_eval(vec2 c, vec2 v, float t) { \
        ivec2 size = ivec2(${const int: w}, ${const int: h}); \
        vec2 fsz = vec2(size); \
        vec2 qa = c - t * v, qb = c + (1.0 - t) * v; \
        vec2 sa = textureLod($qfwd, (qa + 0.5) / fsz, 0.0).xy; \
        vec2 sb = textureLod($qbwd, (qb + 0.5) / fsz, 0.0).xy; \
        /* a side that is out of the frame does not count */ \
        float ina = m_in(qa, fsz), inb = m_in(qb, fsz); \
        float q = min(mix(1.0, sa.x, ina), mix(1.0, sb.x, inb)) * max(ina, inb); \
        float still = 0.0; \
        @if (consist || stest == 1 || stest >= 3) { \
        ivec2 ia = clamp(ivec2(floor(qa + 0.5)), ivec2(0), size - 1); \
        ivec2 ib = clamp(ivec2(floor(qb + 0.5)), ivec2(0), size - 1); \
        vec2 fa = texelFetch($fwd, ia, 0).xy, fb = texelFetch($bwd, ib, 0).xy; \
        vec2 da = fa - v, db = fb + v; \
        float differ = max(ina * dot(da, da), inb * dot(db, db)); \
        @if (consist) \
            q *= exp(-differ * ${float: inv_consist}); \
        @if (stest == 3) \
            still = 1.0 - exp(-0.25 * differ); \
        @if (stest == 4) \
            still = max(exp(-4.0 * dot(fa, fa)) * sa.y, exp(-4.0 * dot(fb, fb)) * sb.y); \
        @if (stest == 1) \
            still = max(exp(-4.0 * dot(fa, fa)), exp(-4.0 * dot(fb, fb))); \
        @} \
        @if (stest == 2) \
            still = max(sa.y, sb.y); \
        return vec2(q, still); \
    }

#pragma GLSL /* frame_interp_mix */ \
    vec4 color; \
    { \
    float t = ${dynamic float: t}; \
    vec2 pa = ${pos[0]}, pb = ${pos[1]}; \
    ivec2 size = ivec2(${const int: w}, ${const int: h}); \
    vec2 fsz = vec2(size); \
    vec2 isz = 1.0 / fsz; \
    vec2 P = pa * fsz; \
    vec2 c0 = P - 0.5; \
    ivec2 ip = clamp(ivec2(floor(P)), ivec2(0), size - 1); \
    vec2 vf = texelFetch($fwd, ip, 0).xy, vb = -texelFetch($bwd, ip, 0).xy; \
    vec2 v = vf; \
    float side = 0.0; \
    @if (affine) { \
    /* the model's motion of what is at this pixel at t: x = p + t (M p + c) */ \
    mat2 am = mat2(motion_model[0], motion_model[3], motion_model[1], motion_model[4]); \
    vec2 ao = vec2(motion_model[2], motion_model[5]); \
    mat2 ab = mat2(1.0) + t * am; \
    float adet = ab[0][0] * ab[1][1] - ab[1][0] * ab[0][1]; \
    vec2 ax = P - t * ao; \
    vec2 ap = abs(adet) > 1e-3 ? mat2(ab[1][1], -ab[0][1], -ab[1][0], ab[0][0]) * ax / adet : ax; \
    vec2 va = am * ap + ao; \
    @} \
    @if (pick == 1) { \
    if (t > 0.5) { \
        v = vb; \
        side = 1.0; \
    } \
    @} \
    vec2 ev = m_eval(c0, v, t); \
    @if (pick >= 2) { \
    vec2 eb = m_eval(c0, vb, t); \
    /* equally good: the nearer frame's */ \
    if (eb.x > ev.x + (t > 0.5 ? -0.02 : 0.02)) { \
        v = vb; \
        ev = eb; \
        side = 1.0; \
    } \
    @} \
    float Q = ev.x; \
    @if (neigh > 0) { \
    /* the vectors of both fields at the pixel and around it, each weighed */ \
    /* by its quality and by how well the two frames agree along it */ \
    vec4 gs = vec4(0.0), fs = vec4(0.0); \
    float fw = 0.0; \
    vec2 vs = vec2(0.0); \
    float ws = 0.0, qs = 0.0, v2 = 0.0; \
    for (int k = 0; k < ${const int: ncand}; k++) { \
        ivec2 o = ivec2(0); \
        if (k >= 2) { \
            int j = (k - 2) >> 1; \
            int n = (j & 1) == 0 ? ${const int: neigh} : -${const int: neigh}; \
            o = (j & 2) == 0 ? ivec2(n, 0) : ivec2(0, n); \
        } \
        ivec2 pk = clamp(ip + o, ivec2(0), size - 1); \
        vec2 vk = (k & 1) == 0 ? texelFetch($fwd, pk, 0).xy : -texelFetch($bwd, pk, 0).xy; \
        @if (ncand > 10) { \
        /* no motion, and the smooth block motion */ \
        if (k == 10) \
            vk = vec2(0.0); \
        if (k == 11) { \
            vec2 bp = P / vec2(${const float: bgw}, ${const float: bgh}); \
            vk = 0.5 * (textureLod($blkf, bp, 0.0).xy - textureLod($blkb, bp, 0.0).xy); \
        } \
        @} \
        @if (ncand > 12) { \
        if (k == 12) \
            vk = va; \
        @} \
        vec2 ek = m_eval(c0, vk, t); \
        vec2 ka = c0 - t * vk, kb = c0 + (1.0 - t) * vk; \
        vec4 ca = m_sa((ka + 0.5) * isz), cb = m_sb((kb + 0.5) * isz); \
        float ia = m_in(ka, fsz), ib = m_in(kb, fsz); \
        float ph = ia * ib * abs(m_luma(ca) - m_luma(cb)); \
        float wk = (ek.x + 0.02) * exp(-ph * ph * ${float: inv_sigma}) + 1e-9; \
        float ua = (1.0 - t) * ia, ub = t * ib; \
        vec4 ck = ua + ub > 0.0 ? (ua * ca + ub * cb) / (ua + ub) : mix(ca, cb, t); \
        /* for the fallback, by the agreement alone, less what the static */ \
        /* test rejects */ \
        float wf = exp(-ph * ph * ${float: inv_sigma}) \
                 * (1.0 - ek.y * (1.0 - exp(-4.0 * dot(vk, vk)))) + 1e-9; \
        @if (ncand > 12) { \
        if (k == 12) \
            wf *= ${float: mp->affine}; \
        @} \
        fs += wf * ck; \
        fw += wf; \
        gs += wk * ck; \
        vs += wk * vk; \
        ws += wk; \
        qs += wk * ek.x; \
        v2 += wk * dot(vk, vk); \
    } \
    v = vs / ws; \
    Q = qs / ws; \
    /* where they agree, one sample along their mean instead */ \
    float agree = exp(-max(v2 / ws - dot(v, v), 0.0)); \
    @if (fallback == 1) \
        ev.y = m_eval(c0, v, t).y; \
    @} \
    vec2 qa = c0 - t * v, qb = c0 + (1.0 - t) * v; \
    vec4 za = m_sa(pa), zb = m_sb(pb); \
    vec4 ga, gb; \
    @if (sharp) { \
    ga = ${cubic[0]}((qa + 0.5) * isz); \
    gb = ${cubic[1]}((qb + 0.5) * isz); \
    @} else { \
    ga = m_sa((qa + 0.5) * isz); \
    gb = m_sb((qb + 0.5) * isz); \
    @} \
    float wa = (1.0 - t) * m_in(qa, fsz), wb = t * m_in(qb, fsz); \
    vec4 G = wa + wb > 0.0 ? (wa * ga + wb * gb) / (wa + wb) : mix(ga, gb, t); \
    @if (neigh > 0) \
        G = mix(gs / ws, G, agree); \
    /* fallback */ \
    vec4 F = mix(za, zb, t); \
    vec2 V = vec2(0.0); \
    float still = 0.0, wv = 0.0; \
    float ef = abs(m_luma(za) - m_luma(zb)); \
    @if (fallback == 1) { \
    V = v; \
    still = ev.y; \
    wv = 1.0; \
    @} \
    @if (fallback == 2 || fallback == 3) { \
    vec2 bp = P / vec2(${const float: bgw}, ${const float: bgh}); \
    V = 0.5 * (textureLod($blkf, bp, 0.0).xy - textureLod($blkb, bp, 0.0).xy); \
    still = m_eval(c0, V, t).y; \
    wv = 1.0; \
    @} \
    @if (fallback == 6) { \
    V = va; \
    still = m_eval(c0, V, t).y; \
    wv = 1.0; \
    @} \
    @if ((fallback >= 1 && fallback <= 3) || fallback == 6) { \
    vec4 fa = m_sa((c0 - t * V + 0.5) * isz), fb = m_sb((c0 + (1.0 - t) * V + 0.5) * isz); \
    float ev1 = abs(m_luma(fa) - m_luma(fb)); \
    @if (fallback == 3) \
        wv = exp(-max(ev1 * ev1 - ef * ef, 0.0) * ${float: inv_sigma}); \
    still *= 1.0 - exp(-4.0 * dot(V, V)); \
    wv *= 1.0 - still; \
    /* a sample out of the frame does not count */ \
    float ua = (1.0 - t) * m_in(c0 - t * V, fsz), ub = t * m_in(c0 + (1.0 - t) * V, fsz); \
    F = mix(F, ua + ub > 0.0 ? (ua * fa + ub * fb) / (ua + ub) : mix(fa, fb, t), wv); \
    ef = mix(ef, ev1, wv); \
    @} \
    @if ((fallback == 4 || fallback == 5 || fallback == 7) && neigh > 0) \
        F = (fs + ${float: mp->giveup} * F) / (fw + ${float: mp->giveup}); \
    @if (veto) { \
    float eg = max(abs(m_luma(ga) - m_luma(gb)) - ef, 0.0); \
    Q *= exp(-eg * eg * ${float: inv_veto}); \
    @} \
    color = mix(F, G, Q); \
    @if (show == 1) \
        color = vec4(vec3(Q), 1.0); \
    @if (show == 2) \
        color = G; \
    @if (show == 3) \
        color = F; \
    @if (show == 4) \
        color = vec4(still, 0.5 + V / 64.0, 1.0); \
    @if (show == 5) { \
    float differ = 1.0 - exp(-dot(vf - vb, vf - vb)); \
    color = mix(vec4(vec3(0.3 * m_luma(G)), 1.0), \
                vec4(1.0 - side, 0.0, side, 1.0), differ); \
    @} \
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
    case PL_FRAME_INTERP_GATHER:
    case PL_FRAME_INTERP_MIX:
        me_dir = PL_MOTION_BOTH;
        break;
    case PL_FRAME_INTERP_ALGO_COUNT:
    default:
        PL_ERR(sh, "pl_shader_frame_interp: invalid algorithm %d!",
               (int) params->algo);
        return false;
    }

    const bool mix_affine = params->algo == PL_FRAME_INTERP_MIX &&
                            params->mix.fallback >= 6;
    motion_b_request = params->algo == PL_FRAME_INTERP_GATHER ? 1 :
                       params->algo == PL_FRAME_INTERP_MIX ? 2 : 0;
    motion_mix_request = params->mix;

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
    case PL_FRAME_INTERP_GATHER:
        return frame_interp_gather(sh, input, &out_color);
    case PL_FRAME_INTERP_MIX: {
        struct sh_motion_obj *obj = SH_OBJ_LOOKUP(*input->state, PL_SHADER_OBJ_MOTION,
                                                  struct sh_motion_obj);
        if (mix_affine && !motion_model_fit(input->gpu, input->dispatch, obj))
            return false;
        return frame_interp_mix(sh, input, &params->mix, &out_color);
    }
    case PL_FRAME_INTERP_ALGO_COUNT:
        break;
    }

    pl_unreachable(); // rejected above
}
