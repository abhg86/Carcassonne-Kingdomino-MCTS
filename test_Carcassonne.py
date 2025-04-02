import random  
from typing import Optional  
import time
  
from wingedsheep.carcassonne.carcassonne_game import CarcassonneGame  
from wingedsheep.carcassonne.carcassonne_game_state import CarcassonneGameState  
from wingedsheep.carcassonne.objects.actions.action import Action  
from wingedsheep.carcassonne.tile_sets.supplementary_rules import SupplementaryRule  
from wingedsheep.carcassonne.tile_sets.tile_sets import TileSet  
   
game = CarcassonneGame(  
	players=2,  
	tile_sets=[TileSet.BASE],  
	supplementary_rules=[]  
)  
  
start = time.time()
while not game.is_finished():  
    player: int = game.get_current_player()  
    valid_actions: [Action] = game.get_possible_actions()  
    action: Optional[Action] = random.choice(valid_actions)  
    if action is not None:  
        game.step(player, action)  
    # game.render() 

print(time.time() - start)
