import datetime


class TrainingDataGenerator():
    @staticmethod
    def run(options):
        options = {
            'goal_lat_range': [

            ],
            'goal_lon_range': [

            ],
            'goal_time_range': [
                datetime.datetime(2022, 4, 1, 0, 0, 0, tzinfo=datetime.timezone.utc),
                datetime.datetime(2022, 6, 1, 0, 0, 0, tzinfo=datetime.timezone.utc)
            ]
        }