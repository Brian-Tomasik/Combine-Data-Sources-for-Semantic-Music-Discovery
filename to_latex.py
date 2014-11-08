# Convert results pickles to LaTeX tables.

from __future__ import division
from decimal import *
import sys
import re
import pdb
from optparse import OptionParser
import cPickle
import crossvalidate

__author__ = "Brian Tomasik"
__date__ = "May 2009"

#STATS_TO_SHOW = ["AUC", "MAP", "MAP/Baseline", "R-Prec", "10-Prec"]
STATS_TO_SHOW = ["AUC", "MAP", "R-Prec", "10-Prec"]
N_TABLE_COLS = len(STATS_TO_SHOW) + 1

class LatexWriter(object):
    """
    A LatexWriter class is just a shell for running functions to generate
    LaTeX table output.
    """
    def __init__(self):
        """
        No input arguments. The outstring_list acts as a string buffer to hold
        lines that will ultimately be combined as "\n".join(self.outstring_list).
        """
        self.outstring_list = []

    def _write_row(self, item_list, use_newlines=True):
        """
        Given a list of items, write them as a row in the table. Use newlines
        liberally to make the output easier to read.
        """
        str_list = map(lambda item: str(item), item_list)
        if use_newlines:
            self.outstring_list.append(" &\n".join(str_list) + r"\\")
        else:
            self.outstring_list.append(" & ".join(str_list) + r"\\")

    def param_sweep_table(self, per_tagtype_results, fontsize=r"\small", two_cols=True):
        """
        Given an output dictionary, print a corresponding LaTeX table.

        Required params:
        - per_tagtype_results: A dictionary mapping tagtype to results
        dictionaries of parameter sweeps for a given tagtype.
        
        Optional params:
        - fontsize: Default is r"\small". Should be a raw string for some
        valid LaTeX size command.
        - two_cols: Make the table have two columns?
        """
        self.outstring_list.append(r"\begin{table*}[h]")
        self.outstring_list.append(r"\centering")
        self.outstring_list.append(fontsize)
        if two_cols:
            self.outstring_list.append(r"\begin{tabular}{%s}" % ("l|" + "c" * (N_TABLE_COLS-1) + "|" + "c" * (N_TABLE_COLS-1)))
        else:
            self.outstring_list.append(r"\begin{tabular}{%s}" % ("l" + "c" * (N_TABLE_COLS-1)))
        self.outstring_list.append(r"\toprule")
        firstrow = [""]
        firstrow.extend(STATS_TO_SHOW)                                      
        if two_cols:
            firstrow.extend(STATS_TO_SHOW) # Second time....
        self._write_row(firstrow)
        if two_cols:
            (tagtype1, tagtype2) = per_tagtype_results.keys()
            results1 = per_tagtype_results[tagtype1]
            results2 = per_tagtype_results[tagtype2]            
            self._write_two_col_tagtype_subtable(tagtype1, tagtype2, results1, results2)
        else:
            for (tagtype, per_tagtype_results) in per_tagtype_results.iteritems():
                self._write_tagtype_subtable(tagtype, per_tagtype_results)
        self.outstring_list.append(r"\bottomrule")
        self.outstring_list.append(r"\end{tabular}")
        self.outstring_list.append(r"\caption{Results for ??? param sweep.}")
        self.outstring_list.append(r"\label{???}")
        self.outstring_list.append(r"\end{table*}")
        self.outstring_list.append("")
        return "\n".join(self.outstring_list)

    def _write_two_col_tagtype_subtable(self, tagtype1, tagtype2, results1, results2):
        self.outstring_list.append(r"\midrule")
        # Write multicolumn header stuff...
        next_line_list = [""]
        next_line_list.append(r"\multicolumn{%d}{c}{%s}" % (N_TABLE_COLS-1, self._subtable_title(tagtype1, results1)))
        next_line_list.append(r"\multicolumn{%d}{c}{%s}" % (N_TABLE_COLS-1, self._subtable_title(tagtype2, results2)))
        self._write_row(next_line_list)
        # Another \midrule.
        self.outstring_list.append(r"\midrule")
        # We're going to be printing out decimals such that we print at least one significant digit according to the std errors. Therefore, figure out how many digits we need.
        n_decimals_keep1 = self._get_n_decimals_keep_dict(results1)
        n_decimals_keep2 = self._get_n_decimals_keep_dict(results2)
        # Now sort by mean overall AUC of results1 and record the results.
        key_auc_pairs = [(key, results1[key]["overall_avg"]["AUC"][0]) for key in results1.keys()]
        for (key, mean_auc) in sorted(key_auc_pairs, key=lambda (key, mean_auc): mean_auc):
            overall_results1 = results1[key]["overall_avg"]
            overall_results2 = results2[key]["overall_avg"]
            row = ["%s" % self._abbreviate_key(str(key))]
            row.extend(self._list_of_stat_values(overall_results1, n_decimals_keep1))
            row.extend(self._list_of_stat_values(overall_results2, n_decimals_keep2))
            self._write_row(row)

    def _write_tagtype_subtable(self, tagtype, per_param_results):
        self.outstring_list.append(r"\midrule")
        self.outstring_list.append(r"\multicolumn{%d}{c}{%s}\\" % (N_TABLE_COLS, self._subtable_title(tagtype, per_param_results)))
        self.outstring_list.append(r"\midrule")
        # We're going to be printing out decimals such that we print at least one significant digit according to the std errors. Therefore, figure out how many digits we need.
        n_decimals_keep = self._get_n_decimals_keep_dict(per_param_results)
        # Now sort by mean overall AUC and record the results.
        key_auc_pairs = [(key, per_param_results[key]["overall_avg"]["AUC"][0]) for key in per_param_results.keys()]
        for (key, mean_auc) in sorted(key_auc_pairs, key=lambda (key, mean_auc): mean_auc):
            overall_results = per_param_results[key]["overall_avg"]
            row = ["%s" % self._abbreviate_key(str(key))]
            row.extend(self._list_of_stat_values(overall_results, n_decimals_keep))
            self._write_row(row)

    def _list_of_stat_values(self, overall_results, n_decimals_keep):
        list = []
        for stat in STATS_TO_SHOW:
            try:
                (mean, stderr, median, n_tags) = overall_results[stat]
                n_digits = n_decimals_keep[stat]
                val = r"%s$\pm$%s" % (self._str_round(mean, n_digits), self._str_round(stderr, n_digits))
            except KeyError:
                val = "NA"
            list.append(val)
        return list

    def _get_n_decimals_keep_dict(self, per_param_results):
        n_decimals_keep = dict()
        for results in per_param_results.values():
            stat_dict = results["overall_avg"]
            for stat in STATS_TO_SHOW:
                stderr = stat_dict[stat][1]
                n_decimals = -Decimal(str(stderr)).adjusted()
                n_decimals_keep[stat] = max(n_decimals_keep.get(stat,0), n_decimals)
        return n_decimals_keep

    def _str_round(self, num, n_digits):
        return str(Decimal(str(num)).quantize(Decimal(10) ** -n_digits))

    def _subtable_title(self, tagtype, per_param_results):
        example_key = per_param_results.keys()[0]
        per_tag_avg = per_param_results[example_key]["per_tag_avg"]
        n_tags = len(per_tag_avg)
        example_tag = per_tag_avg.keys()[0]
        stats_dict = per_tag_avg[example_tag]
        n_songs = stats_dict["Num Songs"] # not used now...
        if tagtype=="Pandora Genres":
            return "%d Genre Tags" % n_tags
        elif tagtype=="Pandora Acoustic":
            return "%d Acoustic Tags" % n_tags
        else:
            return "unknown tagtype: %s" % tagtype

    def _abbreviate_key(self, keystring):
        """
        Do abbreviations for model types and regtypes.
        """
        abbrev_string = keystring.replace(" and ",r"\&")
        abbrev_string = abbrev_string.replace("Independent","Ind")
        abbrev_string = abbrev_string.replace("Hierarchical","Hier")
        abbrev_string = abbrev_string.replace("Linear","Lin")
        abbrev_string = abbrev_string.replace("Logistic","Log")
        abbrev_string = abbrev_string.replace("Mixture","Mix")
        return abbrev_string

def _parse_options():
    parser = OptionParser()
    parser.add_option('-p', '--pickle', default="overall.pkl",
                      help="""Name of pickle file of overall results.""")
    (options, args) = parser.parse_args()
    return options

def main():
    options = _parse_options()
    pkl_file = open(options.pickle, 'rb')
    per_tagtype_results = cPickle.load(pkl_file)
    pkl_file.close()
    table_str = LatexWriter().param_sweep_table(per_tagtype_results)
    print table_str

if __name__ == "__main__":
    main()
