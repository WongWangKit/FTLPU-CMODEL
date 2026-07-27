#include"VXM.hpp"
#include <fstream>
#include <iomanip>
#include <string>
#include <iostream>
INT32 input_data = -2; // 重置起始值
int main()
{
    VXM vxm;
    // 设置参数（全局生效，因为所有指令共用）
    vxm.set_bias_value(2);
    vxm.set_scale(0.5f);
    vxm.set_zero_point(0);
    // 构造第一条指令：Bias+Convert+RELU，处理5个向量
    VXMInstruction inst1;
    inst1.opcode = (1<<3) | (1<<2) | 1; // 0b1101
    inst1.vector_count = 10;
    vxm.push_instruction(inst1);
    // 构造第二条指令：Bias+Convert+GELU，处理5个向量
    VXMInstruction inst2;
    inst2.opcode = (1<<3) | (1<<2) | 2; // 0b1110
    inst2.vector_count = 5;
    vxm.push_instruction(inst2);

    // 由于 VXM 的 tick 中使用了全局 input_data，我们将其初始化为一个合适值
    // 注意：input_data 是 extern 定义在某个 cpp 中，这里我们直接赋值（需在 VXM.hpp 外部定义）
    // 实际上，最好在 VXM 内部使用输入 FIFO，这里我们暂时保持外部赋值。
    // 为了测试，我们在外部定义 input_data 变量。
    // 由于 input_data 在 VXM.hpp 中声明为 extern，我们需要在 main.cpp 中定义它。
    // 为了简化，我们可以在 main.cpp 中定义 INT32 input_data = 0;
    // 然后 VXM 中会使用它。
    // 我们在这里定义它。
    // 注意：必须在全局作用域定义。
    // 我们在 main 之前定义。

    for(int i=0;i<25;i++)
    {
        vxm.tick();
    }

    // 关闭文件（VXM 析构时会关闭）
    std::cout << "Test finished. Check VXM_test_result.txt" << std::endl;
    return 0;
}