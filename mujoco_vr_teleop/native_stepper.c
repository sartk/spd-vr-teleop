#include <math.h>
#include <stdint.h>

#include <mujoco/mujoco.h>

static const mjtNum kYUpConv[9] = {
    1, 0, 0,
    0, 0, 1,
    0, -1, 0,
};

static void mul33(const mjtNum* a, const mjtNum* b, mjtNum* out) {
    for (int r = 0; r < 3; ++r) {
        for (int c = 0; c < 3; ++c) {
            out[3 * r + c] =
                a[3 * r + 0] * b[3 * 0 + c] +
                a[3 * r + 1] * b[3 * 1 + c] +
                a[3 * r + 2] * b[3 * 2 + c];
        }
    }
}

static void transpose33(const mjtNum* src, mjtNum* dst) {
    dst[0] = src[0];
    dst[1] = src[3];
    dst[2] = src[6];
    dst[3] = src[1];
    dst[4] = src[4];
    dst[5] = src[7];
    dst[6] = src[2];
    dst[7] = src[5];
    dst[8] = src[8];
}

static mjtNum det33(const mjtNum* m) {
    return
        m[0] * (m[4] * m[8] - m[5] * m[7]) -
        m[1] * (m[3] * m[8] - m[5] * m[6]) +
        m[2] * (m[3] * m[7] - m[4] * m[6]);
}

void mjvr_step_model(uintptr_t model_addr, uintptr_t data_addr, int n_substeps) {
    mjModel* model = (mjModel*)(uintptr_t)model_addr;
    mjData* data = (mjData*)(uintptr_t)data_addr;
    if (!model || !data || n_substeps <= 0) {
        return;
    }
    for (int i = 0; i < n_substeps; ++i) {
        mj_step(model, data);
    }
}

void mjvr_pack_body_transforms(
    uintptr_t model_addr,
    uintptr_t data_addr,
    const int* body_ids,
    int body_count,
    float* out_buf
) {
    mjData* data = (mjData*)(uintptr_t)data_addr;
    mjtNum conv_t[9];
    mjtNum tmp[9];
    mjtNum mat_yup[9];
    mjtNum quat_wxyz[4];

    (void)model_addr;

    if (!data || !body_ids || !out_buf || body_count < 0) {
        return;
    }

    transpose33(kYUpConv, conv_t);

    for (int i = 0; i < body_count; ++i) {
        const int body_id = body_ids[i];
        const mjtNum* xpos = data->xpos + 3 * body_id;
        const mjtNum* xmat = data->xmat + 9 * body_id;
        const int offset = i * 8;

        out_buf[offset + 0] = (float)body_id;
        out_buf[offset + 1] = (float)xpos[0];
        out_buf[offset + 2] = (float)xpos[2];
        out_buf[offset + 3] = (float)(-xpos[1]);

        if (fabs((double)det33(xmat)) > 1e-9) {
            mul33(kYUpConv, xmat, tmp);
            mul33(tmp, conv_t, mat_yup);
            mju_mat2Quat(quat_wxyz, mat_yup);
            out_buf[offset + 4] = (float)quat_wxyz[1];
            out_buf[offset + 5] = (float)quat_wxyz[2];
            out_buf[offset + 6] = (float)quat_wxyz[3];
            out_buf[offset + 7] = (float)quat_wxyz[0];
        } else {
            out_buf[offset + 4] = 0.0f;
            out_buf[offset + 5] = 0.0f;
            out_buf[offset + 6] = 0.0f;
            out_buf[offset + 7] = 1.0f;
        }
    }
}
