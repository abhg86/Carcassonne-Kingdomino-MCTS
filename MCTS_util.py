import numpy as np
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.coordinate import Coordinate
from wingedsheep.carcassonne.objects.coordinate_with_side import CoordinateWithSide
from wingedsheep.carcassonne.objects.game_phase import GamePhase
from wingedsheep.carcassonne.objects.meeple_type import MeepleType
from wingedsheep.carcassonne.objects.side import Side
from wingedsheep.carcassonne.objects.tile import Tile

BOARD_SIZE = 35

def to_string(state:CarcassonneGameState):
    return (str(state.board),str(state.placed_meeples), hash(state.next_tile))

def side_to_number(side:Side)->int:
    if side == Side.CENTER:
            return 0
    elif side == Side.TOP:
        return 1
    elif side == Side.RIGHT:
        return 2
    elif side == Side.BOTTOM:
        return 3
    elif side == Side.LEFT:
        return 4

def number_to_side(number:int):
    if number == 0:
        return Side.CENTER
    elif number == 1:
        return Side.TOP
    elif number == 2:
        return Side.RIGHT
    elif number == 3:
        return Side.BOTTOM
    elif number == 4:
        return Side.LEFT

def tile_to_numpy(tile:Tile):
    numpy_tile = np.zeros(5*4)
    if tile.shield:
        numpy_tile[:5] = 1

    if tile.chapel or tile.cathedral:
        numpy_tile[5:10] = 1

    for side in tile.get_city_sides():
        numpy_tile[2*5 + side_to_number(side)] = 1
    if not tile.grass__contains__(Side.CENTER):
        numpy_tile[2*5 + 5] = 1
    
    for side in tile.get_road_ends():
        numpy_tile[3*5 + side_to_number(side)] = 1
    
    return numpy_tile



def board_to_numpy(board:[[Tile]], board_size=BOARD_SIZE):
    numpy_board = np.zeros(board_size*board_size * 5*4)
    for i in range(board_size):
        for j in range(board_size):
            numpy_board[(i*board_size +j) * 5*4 : (i*board_size +j+1) * 5*4] = tile_to_numpy(board[i][j])
    
    return numpy_board

def meeples_to_numpy(placed_meeples:[[MeeplePosition]], board_size=BOARD_SIZE):
    player_nb = len(placed_meeples)
    numpy_meeples = np.zeros(player_nb * board_size*board_size * 5)
    for i,meeples in enumerate(placed_meeples):
        for meeple in meeples:
            row, column = meeple.coordinate_with_side.coordinate.row, meeple.coordinate_with_side.coordinate.column
            side = side_to_number(meeple.coordinate_with_side.side)
            numpy_meeples[i*board_size*board_size*5 + (row*board_size + column)*5 + side] = 1
    
    return numpy_meeples

def player_to_numpy(current_player:int, player_nb:int):
    # Times 5 for visibility
    numpy_player = np.zeros(5*player_nb)
    numpy_player[current_player*5 : (current_player+1)*5]
    return numpy_player

def to_numpy(state:CarcassonneGameState):
    numpy_board = board_to_numpy(state.board)
    numpy_nex_tile = tile_to_numpy(state.next_tile)
    numpy_meeples = board_to_numpy(state.placed_meeples)
    numpy_player = player_to_numpy(state.current_player, state.players)
    return np.concatenate((numpy_board,numpy_nex_tile,numpy_meeples,numpy_player))

def to_hash(state:CarcassonneGameState):
    res = to_numpy(state)
    res = res.tobytes()
    return res




def action_to_number(action:Action):
    if type(action)==PassAction:
        return BOARD_SIZE*BOARD_SIZE*5 + 1
    elif type(action)==MeepleAction:
        row = action.coordinate_with_side.coordinate.row
        column = action.coordinate_with_side.coordinate.column
        side = side_to_number(action.coordinate_with_side.side)
        return column*BOARD_SIZE*5 + row*5 + side
    elif type(action)==TileAction:
        row = action.coordinate.row
        column = action.coordinate.column
        turn = action.tile_rotations
        return column*BOARD_SIZE*5 + row*5 + turn

@staticmethod
def number_to_action(number:int, phase:GamePhase, tile:Tile):
    if number == BOARD_SIZE*BOARD_SIZE*5 + 1:
        return PassAction
    row = number // BOARD_SIZE*5
    column = (number - row *BOARD_SIZE*5) // 5
    side_or_turn = number - row*BOARD_SIZE*5 - column*5       # between 0 and 4 for meeple's sides and 0 and 3 for tile turns
    coord = Coordinate(row, column)
    if phase == GamePhase.MEEPLES:
        coord_w_side = CoordinateWithSide(coord, number_to_side(side_or_turn))
        return MeepleAction(MeepleType.NORMAL, coord_w_side)
    elif phase == GamePhase.TILES:
        return TileAction(tile, coord, side_or_turn)



def score_to_win(scores:[int], terminated:bool):
    # if scores==[0]*len(scores):
    #     return scores
    if terminated:
        winner = np.argmax(scores)
        res = [0]*len(scores)
        res[winner] = 1
        return res
    return [0]*len(scores)