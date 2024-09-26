import sportsdataverse

nfl_df = sportsdataverse.nfl.load_nfl_pbp(seasons=range(1999,2021))

print(nfl_df)