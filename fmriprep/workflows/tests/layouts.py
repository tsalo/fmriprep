from copy import deepcopy


_anat = [
    {'run': 1, 'suffix': 'T1w'},
    {'run': 2, 'suffix': 'T1w'},
    {'suffix': 'T2w'},
]

_func = [
    *(
        {
            'task': 'rest',
            'run': i,
            'suffix': suffix,
            'metadata': {
                'RepetitionTime': 2.0,
                'PhaseEncodingDirection': 'j',
                'TotalReadoutTime': 0.6,
                'EchoTime': 0.03,
                'SliceTiming': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            },
        }
        for suffix in ('bold', 'sbref')
        for i in range(1, 3)
    ),
    *(
        {
            'task': 'nback',
            'echo': i,
            'suffix': 'bold',
            'metadata': {
                'RepetitionTime': 2.0,
                'PhaseEncodingDirection': 'j',
                'TotalReadoutTime': 0.6,
                'EchoTime': 0.015 * i,
                'SliceTiming': [0.0, 0.2, 0.4, 0.6, 0.8, 1.0, 1.2, 1.4, 1.6, 1.8],
            },
        }
        for i in range(1, 4)
    ),
]

_fmap = [
    {'suffix': 'phasediff', 'metadata': {'EchoTime1': 0.005, 'EchoTime2': 0.007}},
    {'suffix': 'magnitude1', 'metadata': {'EchoTime': 0.005}},
    {
        'suffix': 'epi',
        'dir': 'PA',
        'metadata': {'PhaseEncodingDirection': 'j', 'TotalReadoutTime': 0.6},
    },
    {
        'suffix': 'epi',
        'dir': 'AP',
        'metadata': {'PhaseEncodingDirection': 'j-', 'TotalReadoutTime': 0.6},
    },
]

_no_session = {
    '01': {
        'anat': _anat,
        'func': _func,
        'fmap': _fmap,
    },
}

_single_session = {
    '01': [
        {
            'session': 'pre',
            'anat': _anat,
            'func': _func,
            'fmap': _fmap,
        }
    ],
}

_homogeneous_sessions = {
    '01': [
        {
            'session': 'pre',
            'anat': _anat,
            'func': _func,
            'fmap': _fmap,
        },
        {
            'session': 'post',
            'anat': _anat,
            'func': _func,
            'fmap': _fmap,
        },
    ]
}

_heterogeneous_sessions = {
    '01': [
        {
            'session': 'anat',
            'anat': {'suffix': 'T1w'},
        },
        {
            'session': 'fmri',
            'func': [
                {
                    'task': 'rest',
                    'suffix': 'bold',
                    'metadata': {
                        'RepetitionTime': 2.0,
                        'PhaseEncodingDirection': 'j',
                        'TotalReadoutTime': 0.6,
                    },
                }
            ],
        },
    ]
}

LAYOUTS = {
    'no_session': _no_session,
    'single_session': _single_session,
    'homogeneous_sessions': _homogeneous_sessions,
    'heterogeneous_sessions': _heterogeneous_sessions,
}


def get_layout(layout_id: str):
    """Get a layout spec by ID."""
    return deepcopy(LAYOUTS[layout_id])

__all__ = ['get_layout']
