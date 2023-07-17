import cProfile
import pstats
from pstats import SortKey
from traffic import start

create_stats = True
read_stats = True
file_name = "profile_results"

def readStats():
    p = pstats.Stats(file_name)
    p.sort_stats(SortKey.TIME, SortKey.PCALLS).print_stats(15)

if __name__=="__main__":
    if create_stats:
        cProfile.run('start(nogui=True, logging_off=False, convert_to_csv=True, turn_off_rl=False, generate_both_outputs=False)', file_name)

    if read_stats:
        readStats()