import json, os


def configure(sql_host, sql_user, sql_passwd, db_folder):
    """
    This function configures medusa server.

    :param sql_host: sql host
    :type sql_host: string

    :param sql_user: sql user
    :type sql_user: string

    :param sql_passwd: sql password
    :type sql_passwd: string
    """

    # Create folder if not exists
    if not os.path.exists(db_folder):
        os.makedirs(db_folder)

    # Save config json
    config = {
        'sql_host': sql_host,
        'sql_user': sql_user,
        'sql_passwd': sql_passwd,
        'db_folder': db_folder,
    }
    path = os.path.dirname(os.path.abspath(__file__)) + '/config.json'
    with open(path, 'w') as f:
        json.dump(config, f)


def load_config():
    try:
        path = os.path.dirname(os.path.abspath(__file__)) + '/config.json'
        f = open(path, 'r')
        config = json.load(f)
        return config
    except IOError as e:
        raise Exception('Configuration file config.json not found. Use medusa_server configure function')
