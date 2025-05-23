import random
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState
from wingedsheep.carcassonne.objects.coordinate import Coordinate
from wingedsheep.carcassonne.objects.playing_position import PlayingPosition
from wingedsheep.carcassonne.objects.tile import Tile
from wingedsheep.carcassonne.utils.tile_fitter import TileFitter


class TilePositionFinder:

    @staticmethod
    def possible_playing_positions(game_state: CarcassonneGameState, tile_to_play: Tile, nb_max:int=10000) -> [PlayingPosition]:
        if game_state.empty_board():
            return [PlayingPosition(coordinate=game_state.starting_position, turns=0)]

        playing_positions = []
        i=0

        for row_index, board_row in enumerate(game_state.board):
            for column_index, column_tile in enumerate(board_row):
                if column_tile is not None:
                    continue

                if i>=nb_max:
                    return playing_positions

                top = game_state.get_tile(row_index - 1, column_index)
                bottom = game_state.get_tile(row_index + 1, column_index)
                left = game_state.get_tile(row_index, column_index - 1)
                right = game_state.get_tile(row_index, column_index + 1)

                if top is None and right is None and bottom is None and left is None:
                    continue

                for tile_turns in range(0, 4):
                    if TileFitter.fits(tile_to_play.turn(tile_turns), top=top, bottom=bottom, left=left, right=right, game_state=game_state):
                        playing_positions.append(PlayingPosition(coordinate=Coordinate(row=row_index, column=column_index), turns=tile_turns))
                        i+=1

        return playing_positions
    
    @staticmethod
    def random_possible_playing_positions(game_state: CarcassonneGameState, tile_to_play: Tile, nb_max:int=1) -> [PlayingPosition]:
        if game_state.empty_board():
            return [PlayingPosition(coordinate=game_state.starting_position, turns=0)]

        playing_positions = []
        i=0

        rows = list(enumerate(game_state.board))
        random.shuffle(rows)
        for row_index, board_row in rows:
            columns = list(enumerate(board_row))
            random.shuffle(columns)
            for column_index, column_tile in columns:
                if column_tile is not None:
                    continue

                if i>=nb_max:
                    return playing_positions

                top = game_state.get_tile(row_index - 1, column_index)
                bottom = game_state.get_tile(row_index + 1, column_index)
                left = game_state.get_tile(row_index, column_index - 1)
                right = game_state.get_tile(row_index, column_index + 1)

                if top is None and right is None and bottom is None and left is None:
                    continue

                turns=[0,1,2,3]
                random.shuffle(turns)
                for tile_turns in turns:
                    if TileFitter.fits(tile_to_play.turn(tile_turns), top=top, bottom=bottom, left=left, right=right, game_state=game_state):
                        playing_positions.append(PlayingPosition(coordinate=Coordinate(row=row_index, column=column_index), turns=tile_turns))
                        i+=1

        return playing_positions
