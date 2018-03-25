import methods
from fasta_reader import FASTA
import numpy as np
newstr = []
sign = 1
title = ""
def generator(input_file_name, feature_file_name, feature_list):
    # Read fasta format file
    order, sequences = FASTA(input_file_name)

    print ("-> Feature set generating ...");
    for s in order:
        #print s, sequences[s]
        # sign variable is for class
        # 1 for positive and -1 for negative sequences
        # --------------------------------------------------------------------------------------
        if(s[0]=='p'):
            sign = 1
        else:
            sign = -1
        p = sequences[s]

        each_feature_vector = ""
        # Feature 1
        # Frequencey count of A,C,G,T and (G+C) count
        #--------------------------------------------------------------------------------------
        if(feature_list[1]):
            a,c,g,t = methods.frequence_count(p, 'A', 'C', 'G', 'T')
            each_feature_vector = each_feature_vector + "%d," % a + "%d," % c + "%d," % g + "%d," % t + "%d," % (g + c)

        # Feature 2
        # mean, variance and standard-deviation
        # --------------------------------------------------------------------------------------
        if(feature_list[2]):
            a, c, g, t = methods.frequence_count(p, 'A', 'C', 'G', 'T')
            x = [a, c, g, t]
            each_feature_vector = each_feature_vector + "%d," % np.mean(x) + "%d," % np.var(x) + "%d," % np.std(x)

        # Feature 3
        # G-C Skew
        # --------------------------------------------------------------------------------------
        if(feature_list[3]):
            value = 0
            for s in p:
                if s == 'G':
                    value += 1
                elif s == 'C':
                    value += -1
                else:
                    value += 0
                each_feature_vector = each_feature_vector + "%d," % value

            a, c, g, t = methods.frequence_count(p, 'A', 'C', 'G', 'T')
            # AT-GC ratio
            at_gc_ratio = (a + t) / (g + c)
            each_feature_vector = each_feature_vector + "%f," % at_gc_ratio

        # Feature 4
        # K-mar frequency count
        # K = 2,3,4,5,6
        # AA,AC,AG,AT...........................TT
        # AAA,AAC,AAG,AAT......................TTT
        # AAAA,AAAC,AAAG,AAAT.................TTTT
        # AAAAA,AAAAC,AAAAG,AAAAT............TTTTT
        # AAAAAA,AAAAAC,AAAAAG,AAAAAT.......TTTTTT
        # --------------------------------------------------------------------------------------
        if(feature_list[4]):
            each_feature_vector = each_feature_vector + methods.two_mar_frequency_count(p)
            each_feature_vector = each_feature_vector + methods.three_mar_frequency_count(p)
            each_feature_vector = each_feature_vector + methods.four_mar_frequency_count(p)
            each_feature_vector = each_feature_vector + methods.five_mar_frequency_count(p)
            each_feature_vector = each_feature_vector + methods.six_mar_frequency_count(p)


        # Feature 5
        # 2-mar and K-gap count, here k=75
        # A_A,A_C,A_G,A_T......................T_T
        # A__A,A__C,A__G,A__T.................T__T
        # A___A,A___C,A___G,A___T............T___T
        # A____A,A____C,A____G,A____T.......T____T
        # --------------------------------------------------------------------------------------
        if(feature_list[5]):
            for i in range(1,75):
                each_feature_vector = each_feature_vector + methods.two_mar_k_gap(p, i)

        # Feature 6
        # 3-mar and right K-gap
        # AA_A,AA_C,AA_G,AA_T.........................TT_T
        # AA__A,AA__C,AA__G,AA__T....................TT__T
        # AA___A,AA___C,AA___G,AA___T...............TT___T
        # AA____A,AA____C,AA____G,AA____T..........TT____T
        # --------------------------------------------------------------------------------------
        if(feature_list[6]):
            for i in range(1, 75):
                each_feature_vector = each_feature_vector + methods.three_mar_right_k_gap(p, i)


        # Feature 7
        # 3-mar and left K-gap
        # A_AA,A_AC,A_AG,A_AT.........................T_TT
        # A__AA,A__AC,A__AG,A__AT....................T__TT
        # A___AA,A___AC,A___AG,A___AT...............T___TT
        # A____AA,A____AC,A____AG,A____AT..........T____TT
        # Continuer upto 75 gaps
        # --------------------------------------------------------------------------------------
        if(feature_list[7]):
            for i in range(1, 75):
                each_feature_vector = each_feature_vector + methods.three_mar_left_k_gap(p, i)


        # Feature 8
        # Pattern matching with minmum 3 matching is acceptable
        # --------------------------------------------------------------------------------------
        if(feature_list[8]):
            tata1 = "TATAAT"
            tataR = "TAATAT"
            tata2 = "TATAAA"
            tata2R = "AAATAT"
            threshold = 3
            for i in range(6):
                each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(p, tata1, threshold))
                tata1 = tata1[1:] + tata1[:1] # for laft rotation
                each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(p, tataR, threshold))
                tataR = tataR[1:] + tataR[:1] # for laft rotation
                each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(p, tata2, threshold))
                tata2 = tata2[1:] + tata2[:1] # for laft rotation
                each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(p, tata2R, threshold))
                tata2R = tata2R[1:] + tata2R[:1] # for laft rotation

            tata1 = "TTGACA"
            tataR = "ACAGTT"
            for i in range(6):
                each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(p, tata1, threshold))
                tata1 = tata1[1:] + tata1[:1] # for laft rotation
                each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(p, tataR, threshold))
                tataR = tataR[1:] + tataR[:1] # for laft rotation

            each_feature_vector = each_feature_vector + ("%d," % methods.string_matching(p, "AACGAT", threshold))

        # Feature 9
        # Position distance summation of A,C,G,T
        # --------------------------------------------------------------------------------------
        if(feature_list[9]):
            each_feature_vector = each_feature_vector + "%d," % methods.distance_count(p, 'A')
            each_feature_vector = each_feature_vector + "%d," % methods.distance_count(p, 'C')
            each_feature_vector = each_feature_vector + "%d," % methods.distance_count(p, 'G')
            each_feature_vector = each_feature_vector + "%d," % methods.distance_count(p, 'T')



        # Feature 10
        # Dinucleotide Parameters Based on DNasel Digestion Data.
        # https://www.tandfonline.com/doi/abs/10.1080/07391102.1995.10508842
        # "Trinucleotide Models for DNA Bending Propensity: Comparison of Models Based on DNaseI Digestion and Nucleosome Packaging Data"
        # --------------------------------------------------------------------------------------
        if (feature_list[10]):
            each_feature_vector = each_feature_vector + "%f," % methods.dinucleotide_value(p)


        # Numeric value for A=1,C=2,G=3,T=4
        # --------------------------------------------------------------------------------------
        if (feature_list[11]):
            string = p
            temp = temp + methods.numerical_position(string)

        # For Positive and Negative sign
        # the sign variable is class variable for 1 or -1
        # --------------------------------------------------------------------------------------
        each_feature_vector = each_feature_vector+"%d"%sign
        # For combining all Features
        newstr.append(each_feature_vector)

    print ('-> '+feature_file_name +" creating  ...");
    file_object = open(feature_file_name,"w+")
    for p in newstr:
        file_object.writelines(p+"\n")

    file_object.close()
    print ("-> Complete Features Set  ...");