from A0_MCTS import A0_MCTS
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState, GamePhase
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction
from wingedsheep.carcassonne.objects.actions.tile_action import TileAction
from wingedsheep.carcassonne.objects.playing_position import PlayingPosition
from wingedsheep.carcassonne.utils.possible_move_finder import PossibleMoveFinder
from wingedsheep.carcassonne.utils.tile_position_finder import TilePositionFinder


class ActionUtil:

    @staticmethod
    def get_possible_actions(state: CarcassonneGameState, nb_max:int=10000, rdm=False):
        actions: [Action] = []
        if state.phase == GamePhase.TILES:
            if rdm:
                possible_playing_positions: [PlayingPosition] = TilePositionFinder.random_possible_playing_positions(
                    game_state=state,
                    tile_to_play=state.next_tile,
                    nb_max=nb_max
                )
            else:
                possible_playing_positions: [PlayingPosition] = TilePositionFinder.possible_playing_positions(
                    game_state=state,
                    tile_to_play=state.next_tile,
                    nb_max=nb_max
                )
            if len(possible_playing_positions) == 0:
                actions.append(PassAction())
            else:
                playing_position: PlayingPosition
                for playing_position in possible_playing_positions:
                    action = TileAction(
                        tile=state.next_tile.turn(playing_position.turns),
                        coordinate=playing_position.coordinate,
                        tile_rotations=playing_position.turns
                    )
                    actions.append(action)
        elif state.phase == GamePhase.MEEPLES:
            possible_meeple_actions = PossibleMoveFinder.possible_meeple_actions(game_state=state)
            actions.extend(possible_meeple_actions)
            actions.append(PassAction())
        return actions

    @staticmethod
    def getValidMovesMask(state:CarcassonneGameState, ActionSize:int, BOARD_SIZE:int=35, nb_max:int=10000, rdm=False):
        """
        Input:
            state: current state

        Returns:
            validMoves: a binary vector of length ActionSize, 1 for
                        moves that are valid from the current board and player,
                        0 for invalid moves
        """
        actions: [Action] = [0]*ActionSize
        if state.phase == GamePhase.TILES:
            if rdm:
                possible_playing_positions: [PlayingPosition] = TilePositionFinder.random_possible_playing_positions(
                    game_state=state,
                    tile_to_play=state.next_tile,
                    nb_max=nb_max
                )
            else:
                possible_playing_positions: [PlayingPosition] = TilePositionFinder.possible_playing_positions(
                    game_state=state,
                    tile_to_play=state.next_tile,
                    nb_max=nb_max
                )
            if len(possible_playing_positions) == 0:
                actions[-1] = 1
            else:
                playing_position: PlayingPosition
                for playing_position in possible_playing_positions:
                    row = playing_position.coordinate.row
                    column = playing_position.coordinate.column
                    turn = playing_position.turns
                    action_number = column*BOARD_SIZE*5 + row*5 + turn
                    actions[action_number]=1

        elif state.phase == GamePhase.MEEPLES:
            possible_meeple_positions = PossibleMoveFinder.__possible_meeple_positions(game_state=state)
            if state.meeples[state.current_player] > 0:
                for position in possible_meeple_positions:
                    row = position.coordinate.row
                    column = position.coordinate.column
                    side = A0_MCTS.side_to_number(position.side)
                    number = column*BOARD_SIZE*5 + row*5 + side
                    actions[number]=1
            actions[-1]=1
        return actions