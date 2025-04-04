import random  
from typing import Optional  
import time
  
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame  
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState  
from wingedsheep.carcassonne.objects.actions.action import Action
from wingedsheep.carcassonne.objects.actions.meeple_action import MeepleAction
from wingedsheep.carcassonne.objects.actions.pass_action import PassAction  
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule  
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet  
   
  
start = time.time()
for i in range(70):
    game = CarcassonneGame(  
        players=2,  
        tile_sets=[TileSet.BASE],  
        supplementary_rules=[]  
    )  
    while not game.is_finished():  
        player: int = game.get_current_player()  
        valid_actions: [Action] = game.get_possible_actions(nb_max=1,rdm=True)  
        action: Optional[Action] = random.choice(valid_actions)

        if action is not None:  
            game.step(player, action)  
        # game.render()

print(time.time() - start)
