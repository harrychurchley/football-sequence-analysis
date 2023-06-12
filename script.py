import pandas as pd
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import FunctionTransformer
from functions import select_req_data, clean_player_id, split_location_coords, encode_columns, add_type_player, remove_events, map_positions, sort_events, get_leading_events, plot_shot_outcomes, plot_shot_envolvements, get_minutes_played, plot_mins_inv_pos, get_shot_events_list, id_common_event, plot_common_event

all_events = pd.read_csv("data/2_44_1_all_data.csv", dtype={'location': object})

# clean and prep pipeline
pipeline = Pipeline([
    ('select', FunctionTransformer(select_req_data, kw_args={'team': 'Arsenal'})),
    ('clean', FunctionTransformer(clean_player_id)),
    ('split', FunctionTransformer(split_location_coords, kw_args={'drop_originals': True})),
    ('encode', FunctionTransformer(encode_columns, kw_args={'columns': ['type'], 'zero_padding': True})),
    ('add_type_player', FunctionTransformer(add_type_player)),
    ('map_positions', FunctionTransformer(map_positions)),
    ('sort', FunctionTransformer(sort_events))
])

events = pipeline.transform(all_events)
leading_events=get_leading_events(events, 10)
#plot_shot_outcomes(events)
#plot_shot_envolvements(leading_events)
#minutes_played=get_minutes_played(2, 44, "Arsenal")
#plot_mins_inv_pos(events, leading_events, minutes_played)
shot_events=get_shot_events_list(events, n_events=10)
common_event=id_common_event(shot_events,n=0)
plot_common_event(leading_events, common_event)