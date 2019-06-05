import os
import numpy as np
import rpy2.robjects as rn


def r_save_to_disk(file_name, variables_to_save = None):

    if variables_to_save is None:
        save_command = "save(file='%s')" % file_name
    else:
        save_string = ", ".join(variables_to_save)
        save_command = "save(%s, file='%s')" % (save_string, file_name)

    rn.reval(save_command)
    assert(os.path.isfile(file_name))
    return True


def r_clear():
    rn.reval('rm(list=ls());')


### assignment

def r_assign_str(pval, rname):

    assert len(rname) > 0
    assert type(rname) is str

    if type(pval) is str:
        rn.r.assign(rname, pval)
    else:
        rn.r.assign(rname, rn.StrVector(pval))

    return True


def r_assign_float(pval, rname):

    assert len(rname) > 0
    assert type(rname) is str

    if type(pval) is float:
        rn.r.assign(rname, pval)
    else:
        rn.r.assign(rname, rn.FloatVector(pval))

    return True


def r_assign_int(pval, rname):

    assert len(rname) > 0
    assert type(rname) is str

    if type(pval) is int:
        rn.r.assign(rname, pval)
    else:
        rn.r.assign(rname, rn.IntVector(pval))

    return True



def r_assign_empty_list(rname):
    assert len(rname) > 0
    assert type(rname) is str
    rn.reval('%s = list()' % rname)
    return True



def r_assign_value(pval, rname):

    if isinstance(pval, np.ndarray):
        if len(pval) == 0:
            ptype = 'empty'
        else:
            np_type = type(pval[0])
            if np_type == np.dtype('int'):
                ptype = int
            elif np_type == np.dtype('float'):
                ptype = float
            elif np_type == np.dtype('string'):
                ptype = str
    elif isinstance(pval, list):
        if len(pval) == 0:
            ptype = 'empty'
        else:
            if len(pval) > 0:
                ptype = type(pval[0])
                for p in pval:
                    assert type(p) is ptype
    else:
        ptype = type(pval)

    if ptype == 'empty':
        return_value = r_assign_empty_list(rname)
    elif ptype is str:
        return_value = r_assign_str(pval, rname)
    elif ptype is int:
        return_value = r_assign_int(pval, rname)
    elif ptype is float:
        return_value = r_assign_float(pval, rname)
    else:
        return_value = False

    return return_value


def r_assign_list(pdict, rname):

    rn.reval("%s = list();" % rname)

    for key in pdict:
        r_assign_str(key, "field_name")
        r_assign_value(pdict[key], "field_value")
        rn.reval("%s[[field_name]] = field_value" % rname)

    rn.reval("rm(field_name); rm(field_value);")

    return True


def r_assign(pval, rname, print_flag = False):

    assert len(rname) > 0
    assert type(rname) is str

    if type(pval) is dict:
        return_flag = r_assign_list(pval, rname)
    else:
        return_flag = r_assign_value(pval, rname)

    if print_flag:
        print(rn.reval("print(%s)" % rname)[0])

    return return_flag

