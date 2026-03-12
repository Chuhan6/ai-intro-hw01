"""
八皇后求解器单元测试用例
覆盖正常场景、边界场景、异常场景与核心功能验证
"""
import pytest
from src.eight_queens import EightQueensSolver


class TestEightQueensSolver:
    """八皇后求解器测试类"""

    def test_8_queens_solution_count(self):
        """测试核心场景：8皇后问题解的数量为标准值92"""
        solver = EightQueensSolver(n=8)
        solutions = solver.solve()
        assert len(solutions) == 92

    def test_4_queens_solution_count(self):
        """测试正常场景：4皇后问题解的数量为标准值2"""
        solver = EightQueensSolver(n=4)
        solutions = solver.solve()
        assert len(solutions) == 2

    def test_1_queen_boundary(self):
        """测试边界场景：1皇后问题解的数量为1"""
        solver = EightQueensSolver(n=1)
        solutions = solver.solve()
        assert len(solutions) == 1
        assert solutions[0] == [0]

    def test_2_3_queen_boundary(self):
        """测试边界场景：2皇后、3皇后问题无合法解"""
        solver_2 = EightQueensSolver(n=2)
        assert len(solver_2.solve()) == 0

        solver_3 = EightQueensSolver(n=3)
        assert len(solver_3.solve()) == 0

    def test_invalid_n_raise_error(self):
        """测试异常场景：n<1时抛出ValueError"""
        with pytest.raises(ValueError):
            EightQueensSolver(n=0)
        with pytest.raises(ValueError):
            EightQueensSolver(n=-5)

    def test_valid_solution_verify(self):
        """测试解验证功能：合法解返回True"""
        valid_solution_4 = [1, 3, 0, 2]
        valid_solution_8 = [0, 4, 7, 5, 2, 6, 1, 3]
        assert EightQueensSolver.is_valid_solution(valid_solution_4) is True
        assert EightQueensSolver.is_valid_solution(valid_solution_8) is True

    def test_invalid_solution_verify(self):
        """测试解验证功能：非法解返回False"""
        # 列冲突
        invalid_col = [0, 0, 2, 3]
        # 对角线冲突
        invalid_diagonal = [0, 1, 3, 2]
        # 非法列号
        invalid_col_num = [0, 5, 2, 3]
        # 空解
        empty_solution = []

        assert EightQueensSolver.is_valid_solution(invalid_col) is False
        assert EightQueensSolver.is_valid_solution(invalid_diagonal) is False
        assert EightQueensSolver.is_valid_solution(invalid_col_num) is False
        assert EightQueensSolver.is_valid_solution(empty_solution) is False

    def test_board_print_no_error(self):
        """测试棋盘打印功能：合法解无报错执行"""
        solver = EightQueensSolver(n=8)
        solutions = solver.solve()
        # 验证打印方法无异常抛出
        solver.print_board(solutions[0])