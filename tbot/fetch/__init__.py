import tweepy
from ..utils.io_utils import save_json
from typing import List, Optional, Dict, Union

def fetch_user_metadata(
    api : tweepy.API, 
    user_id : Optional[str] = None, 
    user_screen_name : Optional[str] = None,
) -> Optional[tweepy.models.User]:
    '''
    Fetch metadata for single user, either by user_id or user_screen_name (earlier written, more precedence)
    Returns tweepy.models.User, if neither user_id nor user_screen_name provided returns None
    '''

    if user_id is not None:    
        user = api.get_user(user_id = user_id)
    elif user_screen_name is not None:
        user = api.get_user(screen_name = user_screen_name)
    else:
        print('No user_id or screen_name is provided, returning None.')
        user = None
    
    return user

def fetch_users_metadata(
    api : tweepy.API, 
    user_ids : List[str] = [], 
#     user_screen_names : List[str] = [],
    include_entities : bool = True,
    tweet_mode : str = "extended", # "compat", "extended"
    verbose : int = 1,
) -> List[tweepy.models.User]:
    '''
    Fetch metadata for multiple users, by user_ids (List of user_id)
    Returns List of tweepy.models.User's, if provided user_ids is [] then returns []
    '''

    # TODO: add fetch by screen_name
     
    users = []
    num_of_users_per_request = 100
    
    num_of_user_ids = len(user_ids)
    num_of_done_user_ids = 0
    limits_id = list(range(0, num_of_user_ids, num_of_users_per_request)) + [num_of_user_ids]
    batches_id = [user_ids[start:end] for start, end in zip(limits_id[:-1], limits_id[1:])]
    for batch in batches_id:
        if verbose > 0:
            print(f"Fetching users metadata {num_of_done_user_ids}/{num_of_user_ids}")
        try:
            users.extend(api.lookup_users(user_ids = batch, include_entities = include_entities, tweet_mode = tweet_mode))
        except tweepy.error.TweepError as e:
            if verbose:
                print(f"{e}")
        num_of_done_user_ids += len(batch)
        
#     num_of_user_screen_names = len(user_ids)
#     limits_screen_name = list(range(0, num_of_user_screen_names, num_of_users_per_request)) + [num_of_user_screen_names]
#     batches_screen_name = [user_screen_names[start:end] for start, end in zip(limits_screen_name[:-1], limits_screen_name[1:])]
    
#     for batch in batches_screen_name:
#         users.extend(api.lookup_users(screen_names = batch, tweet_mode = 'extended'))
        
    return users

def fetch_user_timeline(
    api : tweepy.API,
    user_id : Optional[str] = None,
    user_screen_name : Optional[str] = None,
    since_id : Optional[str] = None, 
    max_id : Optional[int] = None,
    count : int = 200,
    include_rts : bool = True,
    trim_user : bool = False,
    exclude_replies : bool = False,
    tweet_mode : str = "extended", # 'compat' or 'extended'
    page : int = -1,
    max_number_to_retrieve : int = 200,
) -> List[tweepy.models.Status]:
    '''
    Fetch timeline for single user (max 3200 tweets due to API limitations), either by user_id or user_screen_name (earlier written, more precedence)
    Returns List of tweepy.models.Status'es, if neither user_id nor user_screen_name provided returns None
    '''
    statuses = []
    
    if user_id is not None:
        for status in tweepy.Cursor(
            api.user_timeline, 
            user_id = user_id, 
            since_id = since_id, 
            max_id = max_id, 
            count = count, 
            trim_user = trim_user, 
            exclude_replies = exclude_replies, 
            include_rts = include_rts, 
            page = page, 
            tweet_mode = tweet_mode,
        ).items(max_number_to_retrieve):
            statuses.append(status)
    elif user_screen_name is not None:
        for status in tweepy.Cursor(
            api.user_timeline, 
            screen_name = user_screen_name, 
            since_id = since_id, 
            max_id = max_id, 
            count = count, 
            trim_user = trim_user, 
            exclude_replies = exclude_replies, 
            include_rts = include_rts, 
            page = page, 
            tweet_mode = tweet_mode,
        ).items(max_number_to_retrieve):
            statuses.append(status)
    else:
        print("No user_id or screen_name is provided, returning empty list.")
        
    return statuses

def fetch_users_timeline(
    api : tweepy.API, 
    user_ids : List[str] = [], 
    since_id : Optional[str] = None, 
    max_id : Optional[int] = None,
    count : int = 200,
    include_rts : bool = True,
    trim_user : bool = False,
    exclude_replies : bool = False,
    tweet_mode : str = "extended", # 'compat' or 'extended'
    page : int = -1,
    max_number_to_retrieve : int = 200,
    verbose : int = 1,
) -> List[tweepy.models.Status]:
    all_statuses = []

    for i, user_id in enumerate(user_ids):
        if verbose >= 1:
            print(f"{i+1} / {len(user_ids)}: Fetching timeline of {user_id}")

        user_id_statuses = fetch_user_timeline(api, user_id, None, since_id, max_id, count, include_rts, trim_user, exclude_replies, tweet_mode, page, max_number_to_retrieve)
        all_statuses.extend(user_id_statuses)

    return all_statuses

def fetch_by_search(
    api : tweepy.API,
    q : str,
    geocode : Optional[str] = None,
    lang : Optional[str] = None, # ISO 639-1 code
    locale : Optional[str] = 'ja',
    result_type : Optional[str] = "recent", # among "mixed", "recent", "popular"
    count : Optional[int] = 100,
    until : Optional[str] = None, # in YYYY-MM-DD format, has 7-day limit
    since_id : Optional[str] = None,
    max_id : Optional[str] = None,
    include_entities : bool = True,
) -> List[tweepy.models.SearchResults]:
    '''
    Fetch statuses by query, returns at most 100 statuses for given query
    '''
    result = api.search(
        q = q, 
        geocode = geocode, 
        lang = lang, 
        locale = locale, 
        result_type = result_type, 
        count = count, 
        until = until, 
        since_id = since_id, 
        max_id = max_id, 
        include_entities = include_entities,
    )

    return result

def fetch_user_friends(
    api : tweepy.API, 
    user_id : Optional[str] = None, 
    user_screen_name : Optional[str] = None,
    count : int = 200,
    skip_status : bool = False,
    include_user_entities : bool = True,
    max_number_to_retrieve : int = 200,
) -> List[tweepy.models.User]:

    users = []
    if user_id is not None:
        for user in tweepy.Cursor(api.friends, user_id = user_id, count = count, skip_status = skip_status, include_user_entities = include_user_entities).items(max_number_to_retrieve):
            users.append(user)
    elif user_screen_name is not None:
        for user in tweepy.Cursor(api.friends, screen_name = user_screen_name, count = count, skip_status = skip_status, include_user_entities = include_user_entities).items(max_number_to_retrieve):
            users.append(user)
    else:
        print("No user_id or screen_name is provided, returning empty list.")

    return users

def fetch_user_friends_ids(
    api : tweepy.API, 
    user_id : Optional[str] = None, 
    user_screen_name : Optional[str] = None,
    stringify_ids : bool = True,
    count : int = 5000,
    max_number_to_retrieve : int = 200,
) -> Union[List[str], List[int]]:

    users_ids = []
    if user_id is not None:
        for user in tweepy.Cursor(api.friends_ids, user_id = user_id, stringify_ids = stringify_ids, count = count).items(max_number_to_retrieve):
            users_ids.append(user)
    elif user_screen_name is not None:
        for user in tweepy.Cursor(api.friends_ids, screen_name = user_screen_name, stringify_ids = stringify_ids, count = count).items(max_number_to_retrieve):
            users_ids.append(user)
    else:
        print("No user_id or screen_name is provided, returning empty list.")

    return users_ids
    
def fetch_user_followers(
    api : tweepy.API, 
    user_id : Optional[str] = None, 
    user_screen_name : Optional[str] = None,
    count : int = 200,
    skip_status : bool = False,
    include_user_entities : bool = True,
    max_number_to_retrieve : int = 200,
) -> List[tweepy.models.User]:

    users = []
    if user_id is not None:
        for user in tweepy.Cursor(api.followers, user_id = user_id, count = count, skip_status = skip_status, include_user_entities = include_user_entities).items(max_number_to_retrieve):
            users.append(user)
    elif user_screen_name is not None:
        for user in tweepy.Cursor(api.followers, screen_name = user_screen_name, count = count, skip_status = skip_status, include_user_entities = include_user_entities).items(max_number_to_retrieve):
            users.append(user)
    else:
        print("No user_id or screen_name is provided, returning empty list.")

    return users

def fetch_user_followers_ids(
    api : tweepy.API, 
    user_id : Optional[str] = None, 
    user_screen_name : Optional[str] = None,
    stringify_ids : bool = True,
    count : int = 5000,
    max_number_to_retrieve : int = 200,
) -> Union[List[str], List[int]]:

    users_ids = []
    if user_id is not None:
        for user in tweepy.Cursor(api.followers_ids, user_id = user_id, stringify_ids = stringify_ids, count = count).items(max_number_to_retrieve):
            users_ids.append(user)
    elif user_screen_name is not None:
        for user in tweepy.Cursor(api.followers_ids, screen_name = user_screen_name, stringify_ids = stringify_ids, count = count).items(max_number_to_retrieve):
            users_ids.append(user)
    else:
        print("No user_id or screen_name is provided, returning empty list.")

    return users_ids

def fetch_users_connectivity(
    api : tweepy.API, 
    user_ids : List[str] = [], 
    stringify_ids : bool = True,
    count : int = 5000,
    max_number_to_retrieve : int = 200,
    include_entities : bool = True,
    tweet_mode : str = "extended", # "compat", "extended"
    verbose : int = 1,
) -> Dict[str, Dict[str, List[tweepy.models.User]]]:

    res = {}

    for i, user_id in enumerate(user_ids):
        if verbose >= 1:
            print(f"{i+1} / {len(user_ids)}: Fetching connections of {user_id}")

        if verbose >= 1:
            print("Fetching friends' ids")
        friends_ids = fetch_user_friends_ids(api, user_id, None, stringify_ids, count, max_number_to_retrieve)
        if verbose >= 1:
            print("Fetching followers' ids")
        followers_ids = fetch_user_followers_ids(api, user_id, None, stringify_ids, count, max_number_to_retrieve)

        if verbose >= 1:
            print("Fetching friends' metadata")
        friends = fetch_users_metadata(api, friends_ids, include_entities, tweet_mode, verbose)
        if verbose >= 1:
            print("Fetching followers' metadata")
        followers = fetch_users_metadata(api, followers_ids, include_entities, tweet_mode, verbose)
        
        res[user_id] = {"friends": friends, "followers": followers}

    return res

def fetch_user_all_data(
    api : tweepy.API,
    user_id : Optional[str] = None,
    user_screen_name : Optional[str] = None,
    is_verbose : bool = False,
    is_save_json : bool = True,
    is_return : bool = True,
) -> Optional[Dict[str, Union[str, Dict, List]]]:

    if user_id or user_screen_name:
        # fetch user metadata
        user_metadata = fetch_user_metadata(api, user_id, user_screen_name)

        # update user_id and user_screen_name with data fetched from API, in case it was not provided etc.
        user_id = user_metadata.id_str
        user_screen_name = user_metadata.screen_name

        # fetch user timeline
        user_timeline = fetch_user_timeline(api, user_id, user_screen_name)

        # fetch mentions to user
        user_mentions = fetch_by_search(api, q=f'(to:{user_screen_name})') # also contains user's mentions to him/herself
        user_mentions = list(filter(lambda x: x.user.id_str != user_id, user_mentions))
        # user_self_mentions = list(filter(lambda x: x.user.id_str == user_id, user_mentions))

        # generate json file
        user_json = {
            'id_str' : user_id,
            'screen_name' : user_screen_name,
            'user_metadata' : user_metadata._json,
            'user_timeline' : list(map(lambda x: x._json, user_timeline)),
            'user_mentions' : list(map(lambda x: x._json, user_mentions)),
        }

        if is_save_json:
            save_json(f'data/{user_id}_{user_screen_name}_all.json', user_json)

        if is_return:
            return user_json
        else:
            return

    else:
        print('No user_id or screen_name is provided')
        return

def fetch_users_profile_images(
    target_folder_path : str,
    user_ids : List[str] = [],
    urls : List[str] = [],
) -> None:
    # TODO: implement the function
    for url in urls:
        pass

def fetch_users_profile_banners(
    target_folder_path : str,
    user_ids : List[str] = [],
    urls : List[str] = [],
) -> None:
    # TODO: implement the function
    for url in urls:
        pass