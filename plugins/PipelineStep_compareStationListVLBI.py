from lofarpipe.support.data_map import DataMap, DataProduct
import pyrap.tables as pt
from losoto.h5parm import h5parm
import logging

def plugin_main(args, **kwargs):
    """
    Takes in list of targets and an h5parm solution set and returns a list of stations
    in the target data which mismatch the calibrator solutions antenna table
    
    Parameters
    ----------
    mapfile_in : str
        Mapfile for input measurement sets
    h5parmdb: str
        Location of the solution h5parm set
    filter: str
        Default filter constrains for the ndppp_prep_target step (usually removing International Baselines)
    
    Returns
    -------
    result : dict
        Output station names to filter
    """
    mapfile_in     = kwargs['mapfile_in']
    h5parmdb       = kwargs['h5parmdb']
    solset_name    = kwargs['solset_name']
    filter         = kwargs['filter']
    data           = DataMap.load(mapfile_in)
    mslist         = [data[i].file for i in xrange(len(data))]
    
    if len(mslist) == 0:
        raise ValueError("Did not find any existing directory in input MS list!")
        pass
    else:
        MS = mslist[0]
        pass
    
    ## reading ANTENNA table of MS
    logging.info('Collecting information from the ANTENNA table.')
    antennaTable = pt.table(MS + "::ANTENNA", ack = False)
    antennaNames = antennaTable.getcol('NAME')

    if solset_name == 'vlbi':
        ## reading in h5parm
        data   = h5parm(h5parmdb, readonly = True)   
        ## reading ANTENNA information from target / phase
        target = data.getSolset('target')
        names = target.getSoltabNames()
        phstab = [ xx for xx in names if 'RMextract' not in xx ][0]
        soltab = target.getSoltab(phstab)
        phsants = soltab.getAxisValues('ant')
        dutch_ants = [ xx for xx in phsants if 'CS' in xx or 'RS' in xx ]
        ## reading ANTENNA information from calibrator
        solset = data.getSolset('calibrator')
        station_names = solset.getAnt().keys()
        int_ants = [ xx for xx in station_names if 'CS' not in xx and 'RS' not in xx ]
        cal_dutch = [ xx for xx in station_names if 'CS' in xx or 'RS' in xx ]
        ## remove core/remote stations not present for calibrator
        all_dutch_ants = [ xx for xx in dutch_ants if xx in cal_dutch ]
        station_names = all_dutch_ants + int_ants
    else:
        ## reading ANTENNA information of h5parm
        data   = h5parm(h5parmdb, readonly = True)
        solset = data.getSolset(solset_name)
        station_names = solset.getAnt().keys()
    ## check whether there are more stations in the target than in the calibrator solutions
    missing_stations = list(set(antennaNames) - set(station_names))
    for missing_station in missing_stations:
        filter += ';!' + missing_station + '*'
        pass

    #filter = filter.lstrip(';')

    data.close()
    
    ## return results
    result = {'filter':str(filter)}
    return(result)
    
    pass    
