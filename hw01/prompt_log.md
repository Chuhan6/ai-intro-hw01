# AI工具全流程交互日志
项目：基于AI协作的八皇后问题工程化实践
AI工具：Claude Code、Cursor

## 阶段1：Python标准工程初始化
### 需求描述Prompt
我需要创建标准 Python 工程项目，根目录 hw01，包含 src（核心代码）、tests（测试代码）文件夹，src/tests 下有__init__.py，根目录有 requirements.txt（依赖 pytest），所有文件符合 PEP8 规范。

### AI输出结果
1. 生成符合Python工程规范的目录结构
2. 生成src/__init__.py、tests/__init__.py初始内容
3. 生成requirements.txt指定pytest依赖

## 阶段2：八皇后问题求解器核心实现
### 需求描述Prompt
实现八皇后求解器类 EightQueensSolver，支持自定义棋盘大小，solve () 返回所有合法解，is_valid_solution () 验证解合法性，print_board () 打印可视化棋盘，处理 n<1 的异常，保证 8 皇后解数量为 92 个。

### AI输出结果
1. 生成完整的EightQueensSolver类，含回溯法核心逻辑
2. 补充类型注解和文档字符串，符合PEP8规范
3. 处理边界与异常场景，保证解数量正确

## 阶段3：单元测试用例编写
### 需求描述Prompt
基于 EightQueensSolver 编写单元测试，覆盖正常场景（8 皇后 92 解、4 皇后 2 解）、边界场景（1 皇后 1 解、2/3 皇后 0 解）、异常场景（n<1 抛异常）、解验证、棋盘打印。

### AI输出结果
1. 生成完整的TestEightQueensSolver测试类
2. 覆盖所有要求的测试场景，用例独立可运行

## 阶段4：Bug引入与AI定位修复
### Bug1：列冲突判断逻辑错误
#### 问题代码
```python
def _is_conflict(self, queens: List[int], row: int, col: int) -> bool:
    for i in range(row):
        if row == queens[i]:  # 错误：行号对比列号
            return True
        if abs(row - i) == abs(col - queens[i]):
            return True
    return False
    列冲突判断错误，导致8皇后解数量远超92，分析原因并修复。
    修复结果
将row == queens[i]改为col == queens[i]，正确判断列冲突。
Bug2：边界条件处理错误
问题代码
python
运行
def __init__(self, n: int = 8):
    if n <= 1:  # 错误：n=1被判定为非法
        raise ValueError("棋盘大小n必须为大于等于1的正整数")
    self.n = n
    self.solutions: List[List[int]] = []
修复 Prompt
plaintext
n=1时抛出异常，1皇后应返回1个解，分析原因并修复。
修复结果
将n <= 1改为n < 1，仅过滤 0 和负数输入。
Bug3：对角线冲突判断逻辑写反
问题代码
python
运行
def _is_conflict(self, queens: List[int], row: int, col: int) -> bool:
    for i in range(row):
        if col == queens[i]:
            return True
        if abs(row - i) != abs(col - queens[i]):  # 错误：逻辑写反
            return True
    return False
修复 Prompt
plaintext
n=4/8皇后返回0解，分析对角线冲突判断错误并修复。
修复结果
将!=改为==，正确判断对角线冲突。
plaintext