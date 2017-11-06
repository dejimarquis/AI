from sudoku import Sudoku


class Var:
    def __init__(self):
        self.value = 0
        self.domain = range(1, 10)


class CSP(object):
    """
    This class is used to solve the CSP with backtracking.
    """

    def __init__(self, sudoku):
        self.variables = [[Var() for i in range(9)] for j in range(9)]
        for i in range(9):
            for j in range(9):
                self.variables[i][j].value = sudoku.board[i][j]
                if self.variables[i][j].value == 0:
                    self.variables[i][j].domain = list(range(1, 10))

    def select_unassigned_var(self):
        grid = self.variables
        for i in range(9):
            for j in range(9):
                if grid[i][j].value == 0:
                    return i, j

    def select_unassigned_mrv(self):
        min_size = 10
        min_index = 0, 0

        for a in range(9):
            for b in range(9):
                if self.variables[a][b].value == 0:
                    temp_row = a
                    temp_column = b

                    ## check rows & columns
                    for x in range(9):
                        if self.variables[a][x].value != 0 and self.variables[a][x].value in self.variables[a][b].domain:
                            self.variables[a][b].domain.remove(self.variables[a][x].value)
                        if self.variables[x][b].value != 0 and self.variables[x][b].value in self.variables[a][b].domain:
                            self.variables[a][b].domain.remove(self.variables[x][b].value)

                    ## sub square regions
                    if temp_row % 3 != 0:
                        if (temp_row - 1) % 3 == 0:
                            temp_row -= 1
                        elif (temp_row - 2) % 3 == 0:
                            temp_row -= 2
                    if temp_column % 3 != 0:
                        if (temp_column - 1) % 3 == 0:
                            temp_column -= 1
                        elif (temp_column - 2) % 3 == 0:
                            temp_column -= 2
                    for i in range(3):
                        for j in range(3):
                            if self.variables[temp_row + i][temp_column + j].value != 0 and self.variables[temp_row + i][temp_column + j].value in self.variables[a][b].domain:
                                self.variables[a][b].domain.remove(
                                    self.variables[temp_row + i][temp_column + j].value)

                    if len(self.variables[a][b].domain) < min_size:
                        min_size = len(self.variables[a][b].domain)
                        min_index = a, b

        # restore all other variable's domain apart from min_index
        for t in range(9):
            for y in range(9):
                if (t, y) != min_index and self.variables[t][y].value == 0:
                    self.variables[t][y].domain = list(range(1, 10))

        return min_index

    def consistent(self, row, column, value):
        temp_row = row
        temp_column = column

        ## check rows & columns
        for x in range(9):
            if self.variables[temp_row][x].value == value:
                return False
            if self.variables[x][temp_column].value == value:
                return False

        ## sub square regions
        if temp_row % 3 != 0:
            if (temp_row - 1) % 3 == 0:
                temp_row -= 1
            elif (temp_row - 2) % 3 == 0:
                temp_row -= 2
        if temp_column % 3 != 0:
            if (temp_column - 1) % 3 == 0:
                temp_column -= 1
            elif (temp_column - 2) % 3 == 0:
                temp_column -= 2
        for i in range(3):
            for j in range(3):
                if self.variables[temp_row + i][temp_column + j].value == value:
                    return False
        return True

    def complete(self):
        """
        Tests whether all the tiles in the board are filled in.
        Returns true if the board is filled. False, otherwise.
        """
        for i in range(9):
            for j in range(9):
                if self.variables[i][j].value == 0:
                    return False
        return True


class CSP_Solver(object):
    """
    This class is used to solve the CSP with backtracking.
    """
    def __init__(self, puzzle_file):
        self.sudoku = Sudoku(puzzle_file)

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver = CSP_Solver('puz-001.txt')
    ### solved_board, num_guesses = csp_solver.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):
        """
        Solves the Sudoku CSP and returns a list of lists representation
        of the solved sudoku puzzle as well as the number of guesses
        (assignments) required to solve the problem.
        YOU MUST EDIT THIS FUNCTION!!!!!
        """
        csp = CSP(self.sudoku)
        temp = self.backtracking_search(csp)
        ############ commented block was used to inspect sudoku using the sudoku methods #########
        # self.sudoku.board = temp[0]
        # self.sudoku.write('puz-100-solved.txt')
        # print(self.sudoku.complete())
        # print(self.sudoku.overwritten())
        # print(self.sudoku.board_str())
        return temp

    def backtracking_search(self, csp):
        num_of_guesses = [0]
        return self.recursive_backtracking(csp, num_of_guesses)

    def recursive_backtracking(self, csp, num_of_guesses):
        if csp.complete():
            output = [[0 for i in range(9)] for j in range(9)]
            for i in range(9):
                for j in range(9):
                    output[i][j] = csp.variables[i][j].value
            return output, num_of_guesses[0]

        row, column = csp.select_unassigned_var()
        for value in csp.variables[row][column].domain:
            num_of_guesses[0] += 1
            # print(num_of_guesses)
            if csp.consistent(row, column, value):
                csp.variables[row][column].value = value
                result = self.recursive_backtracking(csp, num_of_guesses)
                if result is not None:
                    return result
                csp.variables[row][column].value = 0
        return None


class CSP_Solver_MRV(object):
    """
    This class is used to solve the CSP with backtracking and the MRV
    heuristic.
    """
    def __init__(self, puzzle_file):
        self.sudoku = Sudoku(puzzle_file)

    ################################################################
    ### YOU MUST EDIT THIS FUNCTION!!!!!
    ### We will test your code by constructing a csp_solver instance
    ### e.g.,
    ### csp_solver_mrv = CSP_Solver_MRV('puz-001.txt')
    ### solved_board, num_guesses = csp_solver_mrv.solve()
    ### so your `solve' method must return these two items.
    ################################################################
    def solve(self):
        """
        Solves the Sudoku CSP and returns a list of lists representation
        of the solved sudoku puzzle as well as the number of guesses
        (assignments) required to solve the problem.
        YOU MUST EDIT THIS FUNCTION!!!!!
        """
        csp = CSP(self.sudoku)
        temp = self.backtracking_search(csp)
        ############ commented block was used to inspect sudoku using the sudoku methods #########
        # self.sudoku.board = temp[0]
        # self.sudoku.write('puz-100-solved-mrv.txt')
        # print(self.sudoku.complete())
        # print(self.sudoku.overwritten())
        # print(self.sudoku.board_str())
        return temp

    def backtracking_search(self, csp):
        num_of_guesses = [0]
        depth = 0
        return self.recursive_backtracking(csp, num_of_guesses, depth)

    def recursive_backtracking(self, csp, num_of_guesses, depth):
        if csp.complete():
            output = [[0 for i in range(9)] for j in range(9)]
            for i in range(9):
                for j in range(9):
                    output[i][j] = csp.variables[i][j].value
            return output, num_of_guesses[0]

        row, column = csp.select_unassigned_mrv()
        for value in csp.variables[row][column].domain:
            num_of_guesses[0] += 1
            if csp.consistent(row, column, value):
                csp.variables[row][column].value = value
                result = self.recursive_backtracking(csp, num_of_guesses, depth + 1)
                if result is not None:
                    return result
                csp.variables[row][column].value = 0
        # restore domain if no value is found for that var
        csp.variables[row][column].domain = list(range(1, 10))
        return None

if __name__ == '__main__':
    csp_solver = CSP_Solver('puz-100.txt')
    print(csp_solver.solve())
    csp_solver_mrv = CSP_Solver_MRV('puz-100.txt')
    print(csp_solver_mrv.solve())
