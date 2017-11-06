from eight_puzzle import Puzzle
import time
import queue

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
class Node:
    """
    A class representing a node.
    - 'state' holds the state of the node.
    - 'parent' points to the node's parent.
    - 'action' is the action taken by the parent to produce this node.
    - 'path_cost' is the cost of the path from the root to this node.
    """
    def __init__(self, state, parent, action, path_cost):
        self.state = state
        self.parent = parent
        self.action = action
        self.path_cost = path_cost

    def gen_child(self, problem, action):
        """
        Returns the child node resulting from applying 'action' to this node.
        """
        return Node(state=problem.transitions(self.state, action),
                    parent=self,
                    action=action,
                    path_cost=self.path_cost + problem.step_cost(self.state, action))

    @property
    def state_hashed(self):
        """
        Produces a hashed representation of the node's state for easy
        lookup in a python 'set'.
        """
        return hash(str(self.state))

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
def retrieve_solution(node,num_explored,num_generated):
    """
    Returns the list of actions and the list of states on the
    path to the given goal_state node. Also returns the number
    of nodes explored and generated.
    """
    actions = []
    states = []
    while node.parent is not None:
        actions += [node.action]
        states += [node.state]
        node = node.parent
    states += [node.state]
    return actions[::-1], states[::-1], num_explored, num_generated

##################################################################
### Node class and helper functions provided for your convenience.
### DO NOT EDIT!
##################################################################
def print_solution(solution):
    """
    Prints out the path from the initial state to the goal given
    a tuple of (actions,states) corresponding to the solution.
    """
    actions, states, num_explored, num_generated = solution
    print('Start')
    for step in range(len(actions)):
        print(puzzle.board_str(states[step]))
        print()
        print(actions[step])
        print()
    print('Goal')
    print(puzzle.board_str(states[-1]))
    print()
    print('Number of steps: {:d}'.format(len(actions)))
    print('Nodes explored: {:d}'.format(num_explored))
    print('Nodes generated: {:d}'.format(num_generated))


################################################################
### Skeleton code for your Astar implementation. Fill in here.
################################################################
class Astar:
    """
    A* search.
    - 'problem' is a Puzzle instance.
    """
    def __init__(self, problem):
        self.problem = problem

    def solve(self):
        """
        Perform A* search and return a solution using `retrieve_solution'
        (if a solution exists).
        IMPORTANT: Use node generation time (i.e., time.time()) to split
        ties among nodes with equal f(n).
        """
        ################################################################
        # Your code here.
        ################################################################
        root = Node(puzzle.init_state, None, puzzle.actions(puzzle.init_state), 0)
        q = queue.PriorityQueue()
        qset = set()
        # putting items in the queues in this format [f(n), time, node]
        q.put((self.f(root), 0, root))
        qset.add(root)
        explored = set()
        num_generated = 0

        while not q.empty():
            node = q.get()[2]
            qset.remove(node)
            explored.add(node.state_hashed)

            if puzzle.goal_state == node.state:
                return retrieve_solution(node, len(explored), num_generated)

            for action in puzzle.actions(node.state):
                child = node.gen_child(puzzle, action)
                f_n_child = self.f(child)
                generated_time = time.time()
                node_not_in_q = True
                num_generated += 1

                if child.state_hashed not in explored:
                    for temp_node in qset:
                        # checks if child's state is in queue
                        if temp_node.state_hashed == child.state_hashed:
                            node_not_in_q = False
                            # checks if child has a better f_n
                            if self.f(temp_node) > f_n_child:
                                temp_node.parent = child.parent
                                temp_node.state = child.state
                                temp_node.action = child.action
                                temp_node.path_cost = child.path_cost
                        break
                    if node_not_in_q:
                        q.put((f_n_child, generated_time, child))
                        qset.add(child)

        return None

    def f(self,node):
        '''
        Returns a lower bound estimate on the cost from root through node
        to the goal.
        '''
        return node.path_cost + self.h(node)

    def h(self,node):
        '''
        Returns a lower bound estimate on the cost from node to the goal
        using the Manhattan distance heuristic.
        '''
        ################################################################
        ### Your code here.
        ################################################################

        init_state = node.state
        goal_state = puzzle.goal_state
        return sum(abs(b % 3 - g % 3) + abs(b//3 - g//3)
            for b, g in ((init_state.index(i), goal_state.index(i)) for i in range(1, 9)))

    def branching_factor(self, board, trials=100):
        '''
        Returns an average upper bound for the effective branching factor.
        '''

        b_hi = 0  # average upper bound for branching factor
        for t in range(trials):
            puzzle = Puzzle(board).shuffle()
            solver = Astar(puzzle)
            actions, states, num_explored, num_generated = solver.solve()
            ############################################################
            ### Compute upper bound for branching factor and update b_hi
            ### Your code here.
            ############################################################
            b_hi += (num_generated ** (1/len(actions)))
        return b_hi/trials


if __name__ == '__main__':
    # # Simple puzzle test
    board = [[3,1,2],
             [4,0,5],
             [6,7,8]]

    puzzle = Puzzle(board)
    solver = Astar(puzzle)
    solution = solver.solve()
    print_solution(solution)

    # Harder puzzle test
    board = [[7,2,4],
             [5,0,6],
             [8,3,1]]
    puzzle = Puzzle(board)
    solver = Astar(puzzle)
    solution = solver.solve()
    print(len(solution[0]))

    # branching factor test
    b_hi = solver.branching_factor(board, trials=100)
    print('Upper bound on effective branching factor: {:.2f}'.format(b_hi))
