import os.path

def getPath(session, _su):
    if session == 218:
        if _su != 'SUEP':
            raise ValueError('Only SUEP available for session 218')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/Phase_3/Session_218_EPR_V3DFIS2_01_SUEP/N0c_01/"
    elif session == 256:
        if _su != 'SUEP':
            raise ValueError('Only SUEP available for session 256')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/Phase_5/Session_256_EPR_V3DFIS2_01_SUEP/N0c_01/"
    elif session == 294:
        if _su != 'SUREF':
            raise ValueError('Only SUREF available for session 294')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/Phase_6/Session_294_EPR_V3DFIS2_01_SUREF/N0c_01"
    elif session == 380:
        if _su != 'SUREF':
            raise ValueError('Only SUREF available for session 380')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/Phase_8/Session_380_EPR_V3DFIS2_01_SUREF/N0c_01"
    elif session == 538:
        if _su != 'SUEP':
            raise ValueError('Only SUEP available for session 538')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_538_TSPL_DECSINHR_01_SUEP/N0b_S_01"
    elif session == 550:
        if _su != 'SUEP':
            raise ValueError('Only SUEP available for session 538')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_550_TSPL_DECSINFR_01_SUEP/N0b_S_02"
    elif session == 626:
        if _su != 'SUREF':
            raise ValueError('Only SUREF available for session 626')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_626_TSPL_DECSINHR_01_SUREF/N0b_S_01"
    elif session == 638:
        if _su != 'SUREF':
            raise ValueError('Only SUREF available for session 638')
        _path = "/Users/jberge/science/data/microscope/vol/N0/BDS_3.3.0.3/Techno_3/PHASE_TECHNIQUE/Session_638_TSPL_DECSINFR_01_SUREF/N0b_S_02"
    else:
        raise NotImplementedError()
    if not os.path.isdir(_path):
        raise IOError("No directory " + _path)

    return _path
