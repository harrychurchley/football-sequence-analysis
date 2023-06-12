import ast
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from scipy.stats import rankdata
import matplotlib.pyplot as plt
from statsbombpy import sb
from prefixspan import PrefixSpan
from mplsoccer.pitch import Pitch
from matplotlib.cm import ScalarMappable

def select_req_data(df, team):
    req_cols=['match_id','index','type','team','player','player_id',
            'position','shot_statsbomb_xg', 'shot_outcome','location', 
            'pass_end_location','carry_end_location', 'pass_recipient']
    events=df[req_cols]
    team_events=events[events['team']==team]
    return team_events

def clean_player_id(df):
    df['player_id'] = df['player_id'].fillna('00000') # replace NAs
    df['player_id'] = df['player_id'].astype(str).str.split('.', expand=True)[0] # remove .0
    return df

def parse_location(value):
    if pd.isna(value):
        return np.nan
    else:
        return ast.literal_eval(value)

def split_location_coords(df, drop_originals=False):
    df['location'] = df['location'].apply(parse_location)
    df['pass_end_location'] = df['pass_end_location'].apply(parse_location)
    df['carry_end_location'] = df['carry_end_location'].apply(parse_location)
    
    df['location_x'] = df['location'].str[0]
    df['location_y'] = df['location'].str[1]
    df['pass.end_location_x'] = df['pass_end_location'].str[0]
    df['pass.end_location_y'] = df['pass_end_location'].str[1]
    df['carry.end_location_x'] = df['carry_end_location'].str[0]
    df['carry.end_location_y'] = df['carry_end_location'].str[1]

    # drop orginal cols
    if drop_originals:
        df = df.drop(['location', 'pass_end_location', 'carry_end_location'], axis=1)
    return df

def encode_columns(df, columns, zero_padding=False):
    label_encoder = LabelEncoder()
    for col in columns:
        new_col_name = f"{col}_id"
        encoded_column = label_encoder.fit_transform(df[col])
        if zero_padding:
            num_values = df[col].nunique()
            max_width = len(str(num_values))
            encoded_column_str = [str(val).zfill(max_width) for val in encoded_column]
            df[new_col_name] = encoded_column_str
        else: 
            df[new_col_name] = encoded_column
    return df


def add_type_player(df):
    df["type_player"] = df["type_id"].astype(str) + df["player_id"].astype(str)
    return df

def remove_events(df, events):
    for event in events:
        df = df[df['type'] != event]
    return df

def map_positions(df):
    # map positions to broader positions
    position_mapping = {
        'Right Back': 'Defender',
        'Left Back': 'Defender',
        'Left Midfield': 'Midfielder',
        'Right Midfield': 'Midfielder',
        'Left Center Forward': 'Attacker',
        'Right Center Forward': 'Attacker',
        'Left Defensive Midfield': 'Midfielder',
        'Right Defensive Midfield': 'Midfielder',
        'Left Center Back': 'Defender',
        'Right Center Back': 'Defender',
        'Left Center Midfield': 'Midfielder',
        'Right Center Midfield': 'Midfielder',
        'Goalkeeper': 'Goalkeeper',
        'Center Forward': 'Attacker',
        'Center Defensive Midfield': 'Midfielder',
        'Right Wing': 'Attacker',
        'Left Wing': 'Attacker',
        'Center Attacking Midfield': 'Midfielder',
        'Left Wing Back': 'Defender',
        'Right Attacking Midfield': 'Midfielder',
        'Center Back': 'Defender',
        'Left Attacking Midfield': 'Midfielder',
        'Right Wing Back': 'Defender',
        'Center Midfield': 'Midfielder'
    }
    df['mapped_position'] = df['position'].map(position_mapping)
    return df

def sort_events(df):
    df=df.sort_values(by=['match_id', 'index']).reset_index(drop=True)
    return df

def get_leading_events(df, n_events):
    shot_events = []
    event_indexes = []
    play_outcome = []
    play_xG = []
    for i, row in df.iterrows():
        if row['type'] == 'Shot':
            outcome = row['shot_outcome']
            xG = row['shot_statsbomb_xg']
            start_index = max(0, i - n_events)
            indexes = list(range(start_index, i))
            event_indexes.extend(indexes)
            single_shot_events=[]
            for j in indexes:
                event=df['type_player'][j]
                single_shot_events.append(event)
                play_outcome.append(outcome)
                play_xG.append(xG)
            shot_events.append(single_shot_events)
    leading_events=df.loc[event_indexes]
    leading_events['play_outcome'] = play_outcome 
    leading_events['play_xG'] = play_xG
    leading_events['play_xG_ranked'] = rankdata(play_xG) / (len(play_xG) + 1)
    return leading_events

def plot_shot_outcomes(df):
    # box plot - xG by shot outcome 
    all_shots=df[df['type']=='Shot']
    shot_outcome = all_shots['shot_outcome']
    shot_statsbomb_xg = all_shots['shot_statsbomb_xg']

    # Group the shot_statsbomb_xg data based on shot_outcome
    data = [shot_statsbomb_xg[shot_outcome == outcome] for outcome in shot_outcome.unique()]

    # Create a horizontal box plot with a larger figure size
    plt.figure(figsize=(10, 8))
    box_plot = plt.boxplot(data, labels=shot_outcome.unique(), vert=False)

    # Set labels and title
    plt.xlabel('Expected Goals (xG)')
    plt.ylabel('Shot Outcome')
    plt.title('Box Plot of Shot Outcome vs. Expected Goals')

    # Add counts and mean xG to the plot
    for i, outcome in enumerate(shot_outcome.unique()):
        count = len(data[i])
        mean_xg = np.mean(data[i])
        plt.text(1.01, i + 1, f'n={count}, mean={mean_xg:.2f}', va='center', ha='left')
    # Display the plot
    plt.show()

def get_player_involvements(df):
    involvements=df['player'].value_counts()
    return involvements

def plot_shot_envolvements(df):
    involvements=get_player_involvements(df)
    plt.figure(figsize=(12, 6))
    involvements.plot(kind='bar')
    plt.xlabel('Player')
    plt.ylabel('Count')
    plt.title('Involvements in events leading to shots')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()

def get_mode_positions(df):
    mode_positions=df.groupby('player')['mapped_position'].apply(lambda x: x.mode()[0])
    return mode_positions

def unpack_positions(row):
    if len(row['positions']) > 0:
        pos_data_from = row['positions'][0]
        pos_data_to = row['positions'][-1]
        if pos_data_to['to'] is None:
            pos_data_to['to']='90'
        from_value=int(pos_data_from['from'][:2])
        to_value=int(pos_data_to['to'][:2])
        mins_played=to_value-from_value
        return pd.Series({'from': from_value, 'to': to_value, 'mins_played': mins_played})
    else:
        return pd.Series({'from': 0, 'to': 0, 'mins_played': 0})

def get_minutes_played(competition_id, season_id, team):
    matches = sb.matches(competition_id=competition_id, season_id=season_id)
    team_matches = matches[(matches['home_team'] == team) | (matches['away_team'] == team)]
    
    lineup_data=[]
    for match_id in team_matches['match_id']:
        team_lineup=sb.lineups(match_id)[team]
        team_lineup[['from', 'to', 'mins_played']] = team_lineup.apply(unpack_positions, axis=1)
        lineup_data.append(team_lineup)
    
    all_lineups = pd.concat(lineup_data)
    minutes_played = all_lineups.groupby('player_name')['mins_played'].sum().reset_index()
    minutes_played = minutes_played.rename(columns={'player_name': 'player'})
    return minutes_played

def plot_mins_inv_pos(events, leading_events, minutes_played):
    # get data and merge
    mode_positions=get_mode_positions(events)
    involvements=get_player_involvements(leading_events)
    mins_inv_pos = minutes_played.merge(involvements, on='player').merge(mode_positions, on='player')
    mins_inv_pos = mins_inv_pos.rename(columns={'count': 'involvements'})
    # plot
    label_encoder = LabelEncoder() #Create an instance of LabelEncoder
    encoded_positions = label_encoder.fit_transform(mins_inv_pos['mapped_position']) #Encode the 'position' column
    position_colors = {'Goalkeeper': 'black', 'Defender': 'blue', 'Midfielder': 'green', 'Attacker': 'red'} #Create a dictionary to map positions to colors
    colors = [position_colors[mapped_position] for mapped_position in mins_inv_pos['mapped_position']] #Create a list of colors based on the encoded positions

    plt.figure(figsize=(10, 8))
    plt.scatter(mins_inv_pos['mins_played'], mins_inv_pos['involvements'], c=colors)
    for i, row in mins_inv_pos.iterrows():
        if i % 2 == 0: # Label every 5th point
            plt.annotate(row['player'], (row['mins_played'], row['involvements']), textcoords="offset points", xytext=(5,5), ha='center')
    plt.xlabel('Minutes Played')
    plt.ylabel('Number of Involvements')
    plt.title('Minutes Played vs. Number of Involvements')
    for position, color in position_colors.items():
        plt.scatter([], [], c=color, label=position)
        plt.legend(title='Position', loc='upper left')
        
    plt.show()

def get_shot_events_list(df, n_events):
    shot_events = []
    for i, row in df.iterrows():
        if row['type'] == 'Shot':
            start_index = max(0, i - n_events)  # Calculate the start index, ensuring it doesn't go below 0
            indexes = list(range(start_index, i))
            single_shot_events=[]
            for j in indexes:
                event=df['type_player'][j]
                single_shot_events.append(event)
            shot_events.append(single_shot_events)
    return shot_events

def id_common_event(df, n=0):
    ps = PrefixSpan(df)
    results = ps.frequent(50)
    sorted_results = sorted(results, key=lambda x: x[0], reverse=True)
    sorted_results
    common_event=str(sorted_results[n][1][0])
    return common_event

def plot_common_event(leading_events, common_event, colour_by='play_xG'):
    common_event_data=leading_events[leading_events['type_player']==common_event].reset_index(drop=True)
    type_name=common_event_data.loc[0]['type']
    player_name=common_event_data.loc[0]['player']

    cmap_scheme='summer'
    cmap = plt.cm.get_cmap(cmap_scheme)
    color_by=colour_by
    colors = cmap(common_event_data[color_by])

    pitch = Pitch(pitch_type='statsbomb', pitch_color='#22312b', line_color='#c7d5cc')
    fig, ax = pitch.draw(figsize=(12, 8))

    if type_name=='Ball Receipt*':
        ax.scatter(common_event_data['location_x'], common_event_data['location_y'], c=common_event_data[color_by],
                cmap=cmap_scheme, zorder=10)
    elif type_name=='Pass':
        ax.quiver(common_event_data['location_x'], common_event_data['location_y'], 
                common_event_data['pass.end_location_x'] - common_event_data['location_x'], common_event_data['pass.end_location_y'] - common_event_data['location_y'], 
                color=colors, scale_units='xy', angles='xy', scale=1, width=0.002)
    elif type_name=='Carry': 
        ax.quiver(common_event_data['location_x'], common_event_data['location_y'], 
                common_event_data['carry.end_location_x'] - common_event_data['location_x'], common_event_data['carry.end_location_y'] - common_event_data['location_y'], 
                color=colors, scale_units='xy', angles='xy', scale=1, width=0.002)
    else:
        print("Event must be a ball receipt, carry or pass")
    ax.set_title(f"{player_name} {type_name} map (immediately preceding shots on goal)", fontsize=16, loc='left')
    ax.title.set_position([0.5, 1.05])

    # Add colorbar legend
    sm = ScalarMappable(cmap=cmap)
    sm.set_array(common_event_data[color_by])
    plt.colorbar(sm, ax=ax, shrink=0.8, label="Resulting xG")
    plt.show()