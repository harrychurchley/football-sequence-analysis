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

def map_positions(df)
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