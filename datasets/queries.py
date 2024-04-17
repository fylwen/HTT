from enum import Enum, auto


class BaseQueries(Enum):
    CAMINTR = auto()
    JOINTS3D = auto()
    JOINTS2D = auto()
    IMAGE = auto()
    JOINTSABS25D=auto()

    ACTIONIDX = auto()
    ACTIONNAME = auto()
    VERBIDX=auto()
    OBJIDX=auto()
    CAM2LOCAL=auto()

    TASKACTIONIDX=auto()
    TASKACTIONNAME=auto()
 


class TransQueries(Enum):
    CAMINTR = auto()
    JOINTS3D = auto()
    JOINTS2D = auto()
    IMAGE = auto()
    JITTERMASK = auto()
    SIDE = auto()
    SCALE = auto()
    AFFINETRANS = auto()
    ROTMAT = auto()
    
    JOINTSABS25D=auto()

def one_query_in(candidate_queries, base_queries):
    for query in candidate_queries:
        if query in base_queries:
            return True
    return False


def get_trans_queries(base_queries):
    trans_queries = []
    if BaseQueries.IMAGE in base_queries:
        trans_queries.append(TransQueries.IMAGE)
        trans_queries.append(TransQueries.AFFINETRANS)
        trans_queries.append(TransQueries.ROTMAT)
        trans_queries.append(TransQueries.JITTERMASK)
    if BaseQueries.JOINTS2D in base_queries:
        trans_queries.append(TransQueries.JOINTS2D)
    if BaseQueries.JOINTS3D in base_queries:
        trans_queries.append(TransQueries.JOINTS3D)
    if BaseQueries.CAMINTR in base_queries:
        trans_queries.append(TransQueries.CAMINTR)
    if BaseQueries.JOINTSABS25D in base_queries:
        trans_queries.append(TransQueries.JOINTSABS25D)
    return trans_queries