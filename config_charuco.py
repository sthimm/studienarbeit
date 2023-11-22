import yaml

import yaml


class ChArucoConfig:
    def __init__(self, config_file_path):
        self.config_file_path = config_file_path
        self.board_type = None
        self.row_count = None
        self.col_count = None
        self.square_len = None
        self.marker_len = None
        self.size_x = None
        self.size_y = None

    def load(self):
        with open(self.config_file_path, 'r') as config_file:
            config_data = yaml.safe_load(config_file)
            self.board_type = config_data.get('board_type')
            self.row_count = config_data.get('row_count')
            self.col_count = config_data.get('col_count')
            self.square_len = config_data.get('square_len')
            self.marker_len = config_data.get('marker_len')
            self.size_x = config_data.get('size_x')
            self.size_y = config_data.get('size_y')

    def print(self):
        print()
        print("Charuco Board has the following parameters: ")
        print(80 * "-")
        print("Board type:", self.board_type)
        print("Row count:", self.row_count)
        print("Col count:", self.col_count)
        print("Square len:", self.square_len)
        print("Marker len:", self.marker_len)
        print("Size x:", self.size_x)
        print("Size y:", self.size_y)
        print(80 * "-")
        print()


if __name__ == '__main__':
    config = ChArucoConfig('config_charuco.yaml')
    config.load()
    config.print()
