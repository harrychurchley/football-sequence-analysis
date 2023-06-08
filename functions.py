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

def split_location_coords(df, drop_originals=False): 
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
        encoded_column = label_encoder.fit_transform(events[col])
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
    for i, row in events.iterrows():
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

def plot_shot_envolvements(df):
    involvements=df['player'].value_counts()
    plt.figure(figsize=(12, 6))
    involvements.plot(kind='bar')
    plt.xlabel('Player')
    plt.ylabel('Count')
    plt.title('Involvements in events leading to shots')
    plt.xticks(rotation=90)
    plt.tight_layout()
    plt.show()