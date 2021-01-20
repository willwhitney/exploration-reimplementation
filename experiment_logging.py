import csv
from collections import defaultdict


class AverageMeter(object):
    def __init__(self):
        self._sum = 0
        self._count = 0

    def update(self, value, n=1):
        self._sum += value
        self._count += n

    def value(self):
        return self._sum / max(1, self._count)


class MeterGroup(object):
    def __init__(self, csv_path=None):
        self._meters = defaultdict(AverageMeter)
        self._meter_groups = defaultdict(MeterGroup)
        self._csv_file = None
        if csv_path is not None:
            self._csv_file = open(csv_path, 'w', newline='')
            self._csv_writer = None

    def update(self, meter_path, value):
        split_path = meter_path.split('/')
        head = split_path[0]
        if len(split_path) == 1:
            self._meters[head].update(value)
        else:
            tail = '/'.join(split_path[1:])
            self._meter_groups[head].update(tail, value)

    def value(self, meter_path):
        split_path = meter_path.split('/')
        head = split_path[0]
        if len(split_path) == 1:
            return self._meters[head].value()
        else:
            tail = '/'.join(split_path[1:])
            return self._meter_groups[head].value(tail)

    def values(self):
        values = {}
        for meter_name, meter in self._meters.items():
            values[meter_name] = meter.value()
        for mg_name, mg in self._meter_groups.items():
            for k, v in mg.values().items():
                values[mg_name + '/' + k] = v
        return values

    def write_csv(self):
        if self._csv_file is None:
            for mg in self._meter_groups.values():
                mg.write_csv()
            return

        values = self.values()
        if self._csv_writer is None:
            fields = sorted(values.keys())
            self._csv_writer = csv.DictWriter(
                self._csv_file, fieldnames=fields, restval='NA')
            self._csv_writer.writeheader()
        self._csv_writer.writerow(values)
        self._csv_file.flush()

    def write_console(self, format_str):
        values = self.values()
        print(format_str.format(values=values))

    def clear(self):
        self._meters.clear()
        for mg in self._meter_groups.values():
            mg.clear()


class Logger(MeterGroup):
    # needs to create sub-meter-groups that have csv files
    # and keep track of format strings
    def setup(self, sub_meter_map, summary_format_str=None):
        self.summary_format_str = summary_format_str
        self.sub_meter_map = sub_meter_map
        for k, mg_spec in self.sub_meter_map.items():
            mg = MeterGroup(mg_spec['csv_path'])
            self._meter_groups[k] = mg

    def write_console(self):
        if self.summary_format_str is None:
            for mg_name, mg_spec in self.sub_meter_map.items():
                self._meter_groups[mg_name].write_console(
                    mg_spec['format_str'])
        else:
            super().write_console(self.summary_format_str)

    def write_all(self):
        self.write_console()
        self.write_csv()
        self.clear()


TRAIN_FORMAT_STR = ', '.join((
    "Episode {values[episode]:4.0f}",
    "Train score {values[score]:4.0f}",
    "Train novelty score {values[novelty_score]:4.0f}",))

TEST_FORMAT_STR = ', '.join((
    "Episode {values[episode]:4.0f}",
    "Test score {values[score]:4.0f}",
    "Test novelty score {values[novelty_score]:4.0f}",))

SUMMARY_FORMAT_STR = ', '.join((
    "Episode {values[train/episode]:4.0f}",
    "Train score {values[train/score]:4.0f}",
    "Train nov score {values[train/novelty_score]:4.0f}",
    "Train policy ent {values[train/policy_entropy]:5.2f}",
    "Test score {values[test/score]:4.0f}",
    "Test nov score {values[test/novelty_score]:4.0f}",
    "Test policy ent {values[test/policy_entropy]:5.2f}",
))

default_logger = Logger()


def setup_default_logger(save_dir):
    default_logger.setup({
        'train': {
            'csv_path': f'{save_dir}/train.csv',
            'format_str': TRAIN_FORMAT_STR,
        },
        'test': {
            'csv_path': f'{save_dir}/test.csv',
            'format_str': TEST_FORMAT_STR,
        },
    }, summary_format_str=SUMMARY_FORMAT_STR)


if __name__ == "__main__":
    import random
    default_logger.setup({
        'train': {
            'csv_path': 'derp_train.csv',
            'format_str': TRAIN_FORMAT_STR,
        },
        'test': {
            'csv_path': 'derp_test.csv',
            'format_str': TEST_FORMAT_STR,
        },
    })
    for episode in range(1, 101):
        default_logger.update('train/episode', episode)
        default_logger.update('train/score', random.uniform(0, 100))
        default_logger.update('train/novelty_score', random.uniform(0, 1000))
        default_logger.update('test/episode', episode)
        default_logger.update('test/score', random.uniform(200, 1000))
        default_logger.update('test/novelty_score', random.uniform(0, 100))
        if episode % 10 == 0:
            default_logger.write_all()

