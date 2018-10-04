#!/usr/bin python

#This code takes two file path inputs:
#   1. Path to an XGC input file
#   2. Path to setup.F90
#Using these, it checks that each variable in XGC input file is also found
#in the setup.F90. Useful when reusing an input file that may contain
#old or new tags for the XGC version being used

import sys

def check(input_file,setup_file):
    """Checks that each tag in the input_file is found in the setup_file
    """
    fi = open(input_file,'r')
    fs = open(setup_file,'r')
    slines = fs.read()
    slineslist = slines.split('\n')

    #find all namelist parameters
    #startinds = [m.start() for m in re.finditer('namelist /',slines)]
    startinds = [i for (i,s) in enumerate(slineslist) if 'namelist /' in s]
    pat = r"(?<=namelist.\/).+?(?=\/)"
    namelists = re.findall(pat,slines)
    all_params = {}
    for (si,namelist) in zip(startinds,namelists):
        namelist_params = []; i = 0
        while True:
            line = slineslist[si+i]
            if ('#ifdef' in line) or ('#end' in line) or (line==''): 
                i += 1
                continue
            if 'namelist /' in line: line = line.split('/')[-1]
            params = line.replace(' ','').split(',')
            if '&' in params[-1]: 
                params = params[:-1]
                namelist_params += params
            else:
                namelist_params += params
                break #no more line continutation, break out
            i += 1
        all_params[namelist] = namelist_params

    badparams = []
    badvariables = []
    for line in fi:
        if '&' in line:
            param = line.strip().split('&')[1].split()[0]
            if param not in all_params.keys():
                badparams.append(param)
        elif '=' in line:
            tag = line.split('=')[0]
            if tag not in [x for v in all_params.values() for x in v]:
                badvariables.append(tag)
    print 'param lists NOT found:'
    for badparam in badparams:
        print '\t'+badparam
    print ''
    print 'variables NOT found:'
    for badvariable in badvariables:
        print '\t'+badvariable


if __name__=='__main__':
    input_file = sys.argv[1]
    setup_file = sys.argv[2]

    #TODO: assert valid paths

    check(input_file,setup_file)
