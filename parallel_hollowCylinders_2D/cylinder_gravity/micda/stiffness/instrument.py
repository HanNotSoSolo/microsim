"""
Mechanical characteristics of the instrument
Warning! x0, y0, z0 are in conventional cylindric coordinates (z is the main axis of cylinders), not in MICROSCOPE's convention
"""

#TM masses [kg]
mass = {'SUEP_IS1': 0.401706,
            'SUEP_IS2': 0.300939,
            'SUREF_IS1': 0.401533,
            'SUREF_IS2': 1.359813}

    
# geometry of mechanical parts
tm_cyls = {'IS1-SUEP': {'radius_in': 15.4005e-3, #m
                       'radius_out': 19.695e-3,
                       'height': 43.33e-3, #m
                       'density': 19972, #kg/m^3
                       'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0},
     'IS2-SUEP': {'radius_in': 30.401e-3, #m
                      'radius_out': 34.7005e-3,
                      'height': 79.831e-3, #m
                      'density': 4420, #kg/m^3
                      'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0},
     'IS1-SUREF': {'radius_in': 15.4005e-3, #m
                       'radius_out': 19.695e-3,
                       'height': 43.331e-3, #m
                       'density': 19967, #kg/m^3
                       'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0},
     'IS2-SUREF': {'radius_in': 30.3995e-3, #m
                       'radius_out': 34.6985e-3,
                       'height': 79.821e-3, #m
                       'density': 19980, #kg/m^3
                       'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0}
          }


is1_silica = {'inner': {'radius_in': 9e-3, #m
                            'radius_out': 14.8e-3, #m
                            'height': 78e-3, #m
                            'density': 2329, #kg/m^3
                            'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0},
            'outer': {'radius_in': 20.3e-3, #m
                          'radius_out': 24.85e-3, #m
                          'height': 81e-3, #m
                          'density': 2392, #kg/m^3
                          'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0}
                  }

is2_silica = {'inner': {'radius_in': 25.15e-3, #m
                            'radius_out': 29.8e-3, #m
                            'height': 115e-3, #m
                            'density': 2329, #kg/m^3
                            'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0},
            'outer': {'radius_in': 35.3e-3, #m
                          'radius_out': 40e-3, #m
                          'height': 118e-3, #m
                          'density': 2392, #kg/m^3
                          'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0}
                  }

shield = {'inner': {'radius_in': 43e-3, #m
                            'radius_out': 47e-3, #m
                            'height': 118e-3, #m
                            'density': 8125, #kg/m^3
                            'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0},
            'outer': {'radius_in': 57.5e-3, #m
                          'radius_out': 59.5e-3, #m
                          'height': 118e-3, #m
                          'density': 8125, #kg/m^3
                          'x0': 0, #m
                       'y0': 0, #m
                        'z0': 0},
            'silica base': {'radius_in': 1e-6, #m
                          'radius_out': 59.5e-3, #m
                          'height': 18e-3, #m
                          'density': 2392, #kg/m^3
                          'x0': 0, #m
                       'y0': 0, #m
                        'z0': -68e-3},
            'invar base': {'radius_in': 1e-6, #m
                          'radius_out': 59.5e-3, #m
                          'height': 18e-3, #m
                          'density': 8125, #kg/m^3
                          'x0': 0, #m
                       'y0': 0, #m
                        'z0': -86e-3},
            'upper clamp': {'radius_in': 1e-6, #m
                          'radius_out': 59.5e-3, #m
                          'height': 18e-3, #m
                          'density': 8125, #kg/m^3
                          'x0': 0, #m
                       'y0': 0, #m
                        'z0': 68e-3},
            'vacuum system': {'radius_in': 1e-6, #m
                          'radius_out': 50e-3, #m
                          'height': 60e-3, #m <<<--- ????
                          'density': 2000, #kg/m^3 <<<--- ????
                          'x0': 0, #m
                       'y0': 0, #m
                        'z0': 107e-3}
                    }
