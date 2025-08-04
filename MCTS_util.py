import numpy as np
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.coordinate import Coordinate
from wingedsheep.carcassonne.objects.coordinate_with_side import CoordinateWithSide
from wingedsheep.carcassonne.objects.game_phase import GamePhase
from wingedsheep.carcassonne.objects.meeple_position import MeeplePosition
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.objects.side import Side
from wingedsheep.carcassonne.objects.tile import Tile

BOARD_SIZE = 35

def to_string(state:CarcassonneGameState):
    return (str(state.board),str(state.placed_meeples), hash(state.next_tile))

def side_to_coord(side:Side):
    if side == Side.CENTER:
        return 1,1
    elif side == Side.TOP:
        return 0,1
    elif side == Side.RIGHT:
        return 1,2
    elif side == Side.BOTTOM:
        return 2,1
    elif side == Side.LEFT:
        return 1,0
    elif side == Side.TOP_LEFT:
        return 0,0
    elif side == Side.TOP_RIGHT:
        return 0,2
    elif side == Side.BOTTOM_RIGHT:
        return 2,2
    elif side == Side.BOTTOM_LEFT:
        return 2,0

def coord_to_side(dx:int, dy:int)->Side:
    if dx == 0 and dy == 1:
        return Side.TOP
    elif dx == 1 and dy == 2:
        return Side.RIGHT
    elif dx == 2 and dy == 1:
        return Side.BOTTOM
    elif dx == 1 and dy == 0:
        return Side.LEFT
    elif dx == 0 and dy == 0:
        return Side.TOP_LEFT
    elif dx == 0 and dy == 2:
        return Side.TOP_RIGHT
    elif dx == 2 and dy == 2:
        return Side.BOTTOM_RIGHT
    elif dx == 2 and dy == 0:
        return Side.BOTTOM_LEFT
    elif dx == 1 and dy == 1:
        return Side.CENTER
    else:
        raise ValueError(f"Unknown side: {dx},{dy}")
    
def coord_to_turn(dx:int, dy:int)->int:
    if dx==0 and dy==1:
        return 0
    elif dx==1 and dy==2:
        return 1
    elif dx==2 and dy==1:
        return 2
    elif dx==1 and dy==0:
        return 3
    else:
        raise ValueError(f"Unknown turn: {dx},{dy}")

def turn_to_coord(turn):
    if turn == 0:
        return 0,1
    elif turn == 1:
        return 1,2
    elif turn == 2:
        return 2,1
    elif turn == 3:
        return 1,0
    else:
        raise ValueError(f"Unknown turn: {turn}")

def tile_to_numpy(tile:Tile):
    numpy_tile = np.zeros((5,3,3))

    if tile is None:
        return numpy_tile
    if tile.shield:
        numpy_tile[0,:] = 1

    if tile.chapel or tile.cathedral:
        numpy_tile[1,:] = 1

    for side in tile.get_city_sides():
        x,y = side_to_coord(side)
        numpy_tile[2,x,y] = 1
    if not tile.grass.__contains__(Side.CENTER):
        numpy_tile[2,1,1] = 1
    
    for side in tile.get_road_ends():
        x,y = side_to_coord(side)
        numpy_tile[3,x,y] = 1
    
    for side in tile.grass:
        x,y = side_to_coord(side)
        numpy_tile[4,x,y]
    
    return numpy_tile

def single_tile_to_numpy(tile:Tile, board_size=BOARD_SIZE):
    numpy_tile_repeated = np.zeros((5,board_size*3, board_size*3))
    numpy_tile = tile_to_numpy(tile)
    for i in range(board_size):
        for j in range(board_size):
            numpy_tile_repeated[:,i*3:(i+1)*3, j*3:(j+1)*3] = numpy_tile
    return numpy_tile_repeated

def board_to_numpy(board:[[Tile]], board_size=BOARD_SIZE):
    numpy_board = np.zeros((5,board_size*3,board_size*3))
    for i in range(board_size):
        for j in range(board_size):
            numpy_board[:, i*3:(i+1)*3, j*3:(j+1)*3] = tile_to_numpy(board[i][j])
    
    return numpy_board

def meeples_to_numpy(placed_meeples:[[MeeplePosition]], np_board, board_size=BOARD_SIZE):
    player_nb = len(placed_meeples)
    numpy_meeples = np.zeros((player_nb*4, board_size*3,board_size*3))
    for i,meeples in enumerate(placed_meeples):
        for meeple in meeples:
            row, column = meeple.coordinate_with_side.coordinate.row, meeple.coordinate_with_side.coordinate.column
            side = meeple.coordinate_with_side.side
            dx, dy = side_to_coord(side)
            if meeple.meeple_type == MeepleType.FARMER:
                numpy_meeples[i*4 + 0,row*3 + dx, column*3 + dy] = 1
            elif np_board[2,row*3 + dx, column*3 + dy] == 1:
                # In a city
                numpy_meeples[i*4 + 1,row*3 + dx, column*3 + dy] = 1
            elif np_board[3,row*3 + dx, column*3 + dy] == 1:
                # On a road
                numpy_meeples[i*4 + 2,row*3 + dx, column*3 + dy] = 1
            elif np_board[1,row*3 + dx, column*3 + dy] == 1:
                # In a chapel or cathedral
                numpy_meeples[i*4 + 3,row*3 + dx, column*3 + dy] = 1

    return numpy_meeples

def player_to_numpy(current_player:int, player_nb:int, board_size=BOARD_SIZE):
    # Times 5 for visibility
    numpy_player = np.zeros((player_nb, board_size*3, board_size*3))
    numpy_player[current_player,:,:]
    return numpy_player

def phase_to_numpy(state:CarcassonneGameState, board_size=BOARD_SIZE):
    phase = state.phase
    numpy_phase = np.zeros((2,board_size*3, board_size*3))
    if phase == GamePhase.MEEPLES:
        numpy_phase[0,:,:] = 1
        # Add coords of last tile put on the board
        coord = state.last_tile_action.coordinate
        if coord is not None:
            numpy_phase[1,coord.row*3:coord.row*3+3, coord.column*3:coord.column*3+3] = 1 
    elif phase == GamePhase.TILES:
        numpy_phase[1,:,:] = 1
    return numpy_phase

def to_numpy(state:CarcassonneGameState)->np.ndarray:
    # 5 channels
    numpy_board = board_to_numpy(state.board)
    # 5 channel
    numpy_next_tile = single_tile_to_numpy(state.next_tile)

    res = np.append(numpy_board,numpy_next_tile,0)

    # number of player * 4 channels
    numpy_meeples = meeples_to_numpy(state.placed_meeples, numpy_board, board_size=BOARD_SIZE)

    res = np.append(res, numpy_meeples,0)

    # number of player channels
    numpy_player = player_to_numpy(state.current_player, state.players)

    res = np.append(res, numpy_player,0)

    # 2 channels
    numpy_phase = phase_to_numpy(state)

    # 12 + nb of players * 5 channels
    res = np.append(res,numpy_phase,0)

    return res

def to_hash(state):
    """ 
    Convert a CarcassonneGameState or numpy array to a hashable format.
        :param state: CarcassonneGameState or numpy array
        :return: A hashable representation of the state.
    """
    if isinstance(state, CarcassonneGameState):
        res = to_numpy(state)
        res = res.tobytes()
        return res
    elif isinstance(state, np.ndarray):
        return state.tobytes()
    else:
        raise TypeError("Unsupported type for to_hash")


def action_to_number(action:Action, board_size=BOARD_SIZE):
    if action==PassAction or type(action)==PassAction:
        return board_size*board_size*9                  # no +1 because 0,0 counted
    elif type(action)==MeepleAction:
        row = action.coordinate_with_side.coordinate.row
        column = action.coordinate_with_side.coordinate.column
        dx, dy = side_to_coord(action.coordinate_with_side.side)
        return (row*3+dx)*board_size*3 + column*3 + dy
    elif type(action)==TileAction:
        row = action.coordinate.row
        column = action.coordinate.column
        dx,dy = turn_to_coord(action.tile_rotations)
        return (row*3+dx)*board_size*3 + column*3 + dy
    else:
        raise ValueError(f"Unknown action type: {type(action)}")


def number_to_action(number:int, phase:GamePhase, tile:Tile, board_size=BOARD_SIZE):
    if number == board_size*board_size*9 :              # no +1 because 0,0 counted
        return PassAction()
    row_dx, column_dy = number // (board_size*3), number % (board_size*3)
    row, dx = row_dx // 3, row_dx % 3
    column,dy = column_dy//3, column_dy % 3
    coord = Coordinate(row, column)
    if phase == GamePhase.MEEPLES:
        side = coord_to_side(dx,dy)
        coord_w_side = CoordinateWithSide(coord, side)
        if side in [Side.CENTER, Side.TOP, Side.BOTTOM, Side.LEFT, Side.RIGHT]:
            return MeepleAction(MeepleType.NORMAL, coord_w_side)
        else:
            return MeepleAction(MeepleType.FARMER, coord_w_side)
    elif phase == GamePhase.TILES:
        return TileAction(tile.turn(coord_to_turn(dx,dy)), coord, coord_to_turn(dx,dy))     #needs to turn it here



def score_to_win(scores:[int], terminated:bool):
    # if scores==[0]*len(scores):
    #     return scores
    if terminated:
        best_score = np.max(scores)
        res = -1*np.ones(len(scores))
        winners = np.argwhere(scores==best_score).flatten()
        if len(winners)==1:
            res[winners] = 1
        elif len(winners) > 1:
            # Tie
            res[winners] = 0.1
        return list(res)
    return [0]*len(scores)


class AverageMeter(object):
    """From https://github.com/pytorch/examples/blob/master/imagenet/main.py"""

    def __init__(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def __repr__(self):
        return f'{self.avg:.2e}'

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count


class dotdict(dict):
    def __getattr__(self, name):
        return self[name]
    def __setstate__(self, state):
        self.__dict__ = state
    def __getstate__(self):
        return self.__dict__