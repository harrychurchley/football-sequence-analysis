import pandas as pd
from statsbombpy import sb
from functions import select_req_data, clean_player_id, split_location_coords, encode_columns, add_type_player, remove_events, map_positions, sort_events, get_leading_events, plot_shot_outcomes, get_player_involvements, plot_shot_envolvements, get_mode_positions, unpack_positions, get_minutes_played, plot_mins_inv_pos, get_shot_events_list, id_common_event, plot_common_event

all_events = pd.read_csv("data/2_44_1_all_data.csv", dtype={'location': object})
events = select_req_data(all_events, "Aston Villa")
events = clean_player_id(events)
events = split_location_coords(events, drop_originals=True)

print(events.sample(10))

# my_str=events['location'][10176]
# print(type(my_str))

# str_list = convert_string_to_list(my_str)
# print(str_list)
# print(str_list[0])

#print(events['location'].dtype)