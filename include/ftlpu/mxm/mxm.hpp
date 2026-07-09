#pragma once

#include "ftlpu/mxm/array.hpp"
#include "ftlpu/mxm/control_slice.hpp"
#include "ftlpu/mxm/gemm_engine.hpp"

namespace ftlpu {

class Mxm {
public:
    Mxm()
        : control_(array_)
    {
    }

    Mxm(const Mxm&) = delete;
    Mxm& operator=(const Mxm&) = delete;

    void reset()
    {
        array_.reset();
        control_.reset();
    }

    MxmArray& array()
    {
        return array_;
    }

    const MxmArray& array() const
    {
        return array_;
    }

    MxmControlSlice& control()
    {
        return control_;
    }

    const MxmControlSlice& control() const
    {
        return control_;
    }

    MxmGemmEngine make_gemm_engine() const
    {
        return MxmGemmEngine(array_);
    }

private:
    MxmArray array_{};
    MxmControlSlice control_;
};

} // namespace ftlpu
