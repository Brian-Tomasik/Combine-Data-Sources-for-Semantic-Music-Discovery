# Combiner class

from __future__ import division
import util
import numpy
import rpy
import random
from rpy import r as rc
from math import log, sqrt
import pdb
import sys

__author__ = "Brian Tomasik"
__date__ = "April/May 2009"

# Set numpy printing options; see http://www.scipy.org/Numpy_Example_List_With_Doc#head-cc1302f5e9e57de71b578cf25e8a9ffd8aa3a707
numpy.set_printoptions(precision=2)
numpy.set_printoptions(suppress=True)

ALL_REGRESSIONS = ["Independent Linear", "Hierarchical Linear", "Hierarchical Mixture", "Independent Logistic","Hierarchical Logistic","Product","Sum","Min","Max","Median"]#, "Random Guess"] <-- Random Guess still works, but not needed.
ALL_TAGTYPES = ["Pandora Genres", "Pandora Acoustic", "All", "Last.fm"]
ALL_SOURCES = ["CB", "CF", "WD", "P", "I", "Web Binary", "Web Page Percent", "CF2", "CF3", "CF4", "CF5", "CF6", "Random"]

class Combiner(object):
    """
    A Combiner reads input .tab files relative to the directory "basedir"
    and uses regression to combine them.
    """
    def __init__(self, only_these_tags=None, regtype="Independent Linear", tagtype="All", regmodel="CB and WD and CF and P", verbosity=1, production_run=True, basedir = "../../..", fold_no=None, min_tag_count=1, min_feature_count=0, overwrite_final_tab_file=True, mcmc_reps=15000, max_n_songs="INF", ncomp=2, add_intercept_as_first_feature=True, force_betas_nonneg_except_scrobble_or_if_interactions=True):
        self.regtype = regtype
        self.regmodel = regmodel
        self.tagtype = tagtype
        self._verify_regmodel_written_correctly()
        self.verbosity = verbosity
        self.overwrite_final_tab_file = overwrite_final_tab_file
        self.mcmc_reps = mcmc_reps
        self.max_n_songs = max_n_songs
        self.ncomp = ncomp # only used for "Hierarchical Mixture" regtype
        self.production_run = production_run # If True, disable all debugging shortcuts.
        self.basedir = basedir
        self.fold_no = fold_no
        self.min_tag_count = min_tag_count
        self.min_feature_count = min_feature_count
        self.add_intercept_as_first_feature = add_intercept_as_first_feature
        self.force_betas_nonneg_except_scrobble_or_if_interactions = force_betas_nonneg_except_scrobble_or_if_interactions if "I" not in regmodel else False
        self._read_tag_set(self.tagtype) # get some values for self.only_these_tags
        if only_these_tags is not None:
            self.only_these_tags = only_these_tags.intersection(self.only_these_tags)
        self.beta = dict() # dict: tag -> (dict: source -> (dict: regression statistic -> value)); in particular, the "regression statistics" at least include beta, but possibly also p-values, etc.
        self.stats = dict() # dict: tag -> (dict: stat_type -> value)
        self.best_worst_songs = dict() # dict: tag -> (dict: good/bad -> songid)
        self.songid_to_song = self._get_songid_to_song()
        self._set_or_reset_data_dictionaries()

    def _read_tag_set(self, tagtype):
        self._progress("Reading tags of type %s." % tagtype)
        self.only_these_tags = set()
        if tagtype=="All":
            self._read_pandora_results(just_read_tags=True)
            self._read_lastfm_results(just_read_tags=True)
        elif tagtype=="Last.fm":
            self._read_lastfm_results(just_read_tags=True)
        elif tagtype=="Pandora Genres":
            self._read_pandora_results(just_read_tags=True)
            self.only_these_tags = self.only_these_tags.intersection(self._get_pandora_tag_set("%s/lists/Pandora/PandoraGenreTag.tab" % self.basedir))
        elif tagtype=="Pandora Acoustic":
            self._read_pandora_results(just_read_tags=True)
            self.only_these_tags = self.only_these_tags.intersection(self._get_pandora_tag_set("%s/lists/Pandora/PandoraAcousticTag.tab" % self.basedir))
        else:
            raise ValueError("Bad tagtype = %s." % tagtype)

    def _verify_regmodel_written_correctly(self):
        for source in self.regmodel.split(" and "):
            assert source in ALL_SOURCES, "Invalid regmodel: %s" % self.regmodel

    def _get_songid_to_song(self):
        file = open("%s/lists/songList.tab" % self.basedir)
        songid_dict = dict()
        for line in file:
            (songid_string, artist, song_name, mp3file) = line.rstrip().split("\t")
            songid_dict[int(songid_string)] = "%s, by %s" % (song_name, artist)
        file.close()
        return songid_dict

    def _set_or_reset_data_dictionaries(self):
        """
        Write over old values so that the testing phase can start anew.
        """
        self.only_these_songs = None
        self.all_seen_songs = set()
        self.features = dict()  # dict: tag -> (dict: source -> (dict: songid -> score))
        self.ground_truth = dict()  # dict: tag -> dict of songid -> score
        self.song_lists = dict() # dict: tag -> set of songs used for this tag
        self.sorted_sources = dict() # tag -> ordered list of sources for this X and beta
        self.X = dict() # dict: tag -> X matrix
        self.y = dict() # dict: tag -> X matrix
        self.yhat = dict() # dict: tag -> X matrix

    def _remove_tag(self, tag, verbosity=1):
        if verbosity > 0:
            util.info("\tWARNING: Removing tag = %s." % tag)
        self.only_these_tags.remove(tag)
        if tag in self.features:
            del self.features[tag]
        if tag in self.ground_truth:
            del self.ground_truth[tag]
        if tag in self.song_lists:
            del self.song_lists[tag]
        if tag in self.beta:
            del self.beta[tag]
        if tag in self.stats:
            del self.stats[tag]
        if tag in self.sorted_sources:
            del self.sorted_sources[tag]
        if tag in self.X:
            del self.X[tag]
        if tag in self.y:
            del self.y[tag]
        if tag in self.yhat:
            del self.yhat[tag]

    def nonrare_tags(self, only_these_songs):
        """
        Return the tags that weren't pruned upon reading the input data.
        
        Required params:
        - only_these_songs: A set of the only songs we should consider
        from the input.
        """
        self._progress("Computing nonrare tags.")
        self._read_data(only_these_songs=only_these_songs)
        return self.only_these_tags # Will have been pruned by getting X and y.

    def evaluate_regression(self, training_songs=None, testing_songs=None):
        """
        Get results for a single training-and-testing fold of crossvalidation.
        
        Optional params:
        - training_songs: The set of songs to which to restrict ourselves in
        training. If None, then use no restrictions.
        - testing_songs: The same for testing.
        """
        assert len(training_songs.intersection(testing_songs))==0, "Can't test on training songs!"
        # Training.
        self._read_data(only_these_songs=training_songs) # Will update self.all_seen_songs.
        self._make_X_and_y_matrices()
        self._compute_betas()
        # Testing.
        self._set_or_reset_data_dictionaries()
        self._read_data(only_these_songs=testing_songs)
        self._make_X_and_y_matrices()
        self._compute_yhat()
        self._performance_stats()
        if self.verbosity > 2:
            self._progress(self.stats)
            self._progress(str(self.beta))
        return self.stats

    def fill_in_zeros(self, outfile="final.tab"):
        """
        Generate a final song-score matrix as "outfile" for use by a query
        evaluator.
        
        Optional params:
        - outfile: The file to which to write the final matrix.
        """
        self._read_data(only_tags_have_all_sources=False)
        self._make_X_and_y_matrices()
        self._compute_betas()
        self._compute_yhat()
        self._write_final(outfile=outfile)
        print self.beta

    def _progress(self, message, newline=True):
        """
        Write info if high enough verbosity.
        """
        if self.verbosity > 0:
            util.info(message, newline=newline)

    def _read_data(self, only_these_songs=None, only_tags_have_all_sources=True, seed_random_subset=True):
        """
        """
        self.only_these_songs = only_these_songs
        self._read_features()
        self._read_ground_truth()
        self._remove_uncommon_tags()
        if only_these_songs is not None:
            assert util.is_subset(self.all_seen_songs, only_these_songs), "Input files had songs that shouldn't have been used."
        if self.max_n_songs != "INF":
            if seed_random_subset:
                random.seed("This is just an arbitrary hash with which to seed random. I'm doing this so that we can compare different types of regressions on equal footing, without worrying about differences spuriously introduced by the particular subset that was chosen.")
            self.all_seen_songs = util.random_subset(self.all_seen_songs, self.max_n_songs)
        if only_tags_have_all_sources:
            self._remove_tags_without_all_sources()

    def _read_features(self):
        self._progress("Reading features.")
        if "CF" in self.regmodel or self.regmodel=="Random": # If self.regmodel=="Random", we need to choose some set of tags to use, so just use the same set as CF....
            self._add_source("%s/lists/propogate/propagated_real1.tab" % self.basedir, "CF") # Use this one!
        if "CF2" in self.regmodel:
            self._add_source("%s/lists/propogate/propagated_real2.tab" % self.basedir, "CF2")
        if "CF3" in self.regmodel:
            self._add_source("%s/lists/propogate/propagated_real3.tab" % self.basedir, "CF3")
        if "CF4" in self.regmodel:
            self._add_source("%s/lists/propogate/propagated_binary1.tab" % self.basedir, "CF4")
        if "CF5" in self.regmodel:
            self._add_source("%s/lists/propogate/propagated_binary2.tab" % self.basedir, "CF5")
        if "CF6" in self.regmodel:
            self._add_source("%s/lists/propogate/propagated_binary3.tab" % self.basedir, "CF6")
        if "CB" in self.regmodel:
            if self.fold_no is None:
                self._add_source("%s/lists/autoTags/crossFold/six_five_sec/normalized_all_TSS.tab" % self.basedir, "CB")
            else:
                part_num = self.fold_no+1+2 # Add 1 because the fold numbers are 1-indexed, not 0-indexed. Add another 2 for some unknown reason. Fold 0 corresponds to part3, fold 1 to part4, etc.
                if part_num > 5:
                    part_num -= 5 # change 6 to 1, 7 to 2
                self._add_source("%s/lists/autoTags/crossFold/six_five_sec/normalized_part%d_TSS.tab" % (self.basedir, part_num), "CB")
        if "Web Binary" in self.regmodel:
            self._add_source("%s/lists/web/greppy_binary.tab" % self.basedir, "Web Binary")
        # Take sqrts because count data. Makes slightly more normal. See
        # http://cran.r-project.org/doc/contrib/Faraway-PRA.pdf (p. 84).
        if "Web Page Percent" in self.regmodel:
            self._add_source("%s/lists/web/greppy_pagepct.tab" % self.basedir, "Web Page Percent", transform=lambda x: sqrt(x))
        if "WD" in self.regmodel:
            self._add_source("%s/lists/web/greppy_rawcount.tab" % self.basedir, "WD", transform=lambda x: sqrt(x))
        self._read_scrobble("%s/lists/lastFM/song_scrobbler_data.tab" % self.basedir)
        self._compute_sorted_sources()

    def _add_source(self, filename, sourcename, transform=None, only_a_few_lines=True):
        """
        Update the features dict with data from the file named filename.
        Use the same name for the type of data source.
        """
        self._progress("Adding source = %s" % sourcename)
        file = open(filename, "r")
        line_no = 0
        for line in file:
            line_no += 1
            if line_no % 500 == 0:
                self._progress("cur line = %d" % line_no)
            if only_a_few_lines and not self.production_run and line_no > 200:
                util.info("\tWARNING: stopping at line 200 of input file. Turn off for production runs.")
                break
            cur_tag = self._read_tag(line)
            if cur_tag in self.only_these_tags:
                cur_dict = self._line_to_dict(line.rstrip().split("\t"), transform=transform)
                if cur_dict: # that is, if cur_dict is not empty
                    try:
                        source_dict = self.features[cur_tag]
                    except KeyError:
                        source_dict = dict()
                    try:
                        old_dict = source_dict[sourcename]
                        # If we get here, we need to merge the new
                        # cur_dict with the old one.
                        source_dict[sourcename] = self._merge_song_dicts(old_dict, cur_dict)
                    except KeyError: # We're adding a new source.
                        source_dict[sourcename] = cur_dict
                    self.features[cur_tag] = source_dict
        file.close()

    def _merge_song_dicts(self, dict1, dict2):
        """
        Destructively merge dict1 and dict2, taking the union of song-score
        values and, where songs match, taking the max of the scores.
        """
        # Add the entries of the shorter dict to the longer one. main is
        # longer than other.
        if len(dict1) >= len(dict2):
            main = dict1
            other = dict2
        else:
            main = dict2
            other = dict1
        for (other_key, other_val) in other.iteritems():
            try:
                main_val = main[other_key]
                main[other_key] = max(main_val, other_val)
            except KeyError:
                main[other_key] = other_val
        return main

    def _read_scrobble(self, filename):
        self.log_scrobble_counts = dict()
        self._progress("Reading Scrobble counts.")
        file = open(filename, "r")
        for line in file:
            (songid, score) = map(lambda str: int(str), line.rstrip().split("\t"))
            if self.only_these_songs is None or songid in self.only_these_songs:
                # Add 1+score to prevent log(0).
                self.log_scrobble_counts[songid] = log(score+1)
        file.close()
        self.avg_scrobble = numpy.mean(self.log_scrobble_counts.values())

    def _read_tag(self, line):
        """
        (Force all tags to lower case.)
        """
        (cur_tag, junk, junk2) = line.rstrip().partition("\t")
        return cur_tag.lower()

    def _line_to_dict(self, line_list, transform=None):
        """
        Parse [tag,song,score,song,score,...] to a dictionary.
        If not None, transform is a function that takes input float scalars.
        """
        # tag = line_list[0]
        d = dict()
        for i in range(1,len(line_list)):
            if i % 2 == 1:
                cur_songid = int(line_list[i])
                if self.only_these_songs is None or cur_songid in self.only_these_songs:
                    if transform is None:
                        d[cur_songid] = float(line_list[i+1])
                    else:
                        d[cur_songid] = transform(float(line_list[i+1]))
                    self.all_seen_songs.add(cur_songid)
        return d   

    def _get_pandora_tag_set(self, filename):
        file = open(filename, "r")
        tag_set = set()
        for line in file:
            tag_set.add(line.rstrip().lower()) # Make lower-case!
        file.close()
        return tag_set

    def _read_ground_truth(self):
        self._progress("Reading ground truth.")
        self._read_lastfm_results()
        self._read_pandora_results()
        #self._prune_low_count_tags() See my comment on the function itself for why it's commented out.

    def _read_lastfm_results(self, just_read_tags=False, song_weight=0.66666):
        assert song_weight >= 0 and song_weight <= 1, "Bad song weight."

        # Song results.
        last_fm_song_file = open("%s/lists/lastFM/song_results.tab" % self.basedir, "r")
        for line in last_fm_song_file:
            song_tag = self._read_tag(line)
            if just_read_tags:
                self.only_these_tags.add(song_tag)
            elif song_tag in self.only_these_tags:
                song_dict = self._line_to_dict(line.rstrip().split("\t"))
                assert song_tag not in self.ground_truth, "Duplicate tag in input file"
                self.ground_truth[song_tag] = song_dict
        last_fm_song_file.close()

        # Artist results.
        last_fm_artist_file = open("%s/lists/lastFM/song_artist_results.tab" % self.basedir, "r")
        for line in last_fm_artist_file:
            artist_tag = self._read_tag(line)
            if just_read_tags:
                self.only_these_tags.add(artist_tag)
            elif artist_tag in self.only_these_tags:
                artist_dict = self._line_to_dict(line.rstrip().split("\t"))
                try:
                    new_dict = self.ground_truth[artist_tag]
                    for (songid, artist_score) in artist_dict.items():
                        try:
                            song_score = new_dict[songid]
                            new_dict[songid] = song_weight * song_score + (1-song_weight) * artist_score
                        except KeyError: # song dict doesn't have this songid
                            new_dict[songid] = artist_score
                    self.ground_truth[artist_tag] = new_dict
                except KeyError: # tag doesn't have song info; just use artist info
                    self.ground_truth[artist_tag] = artist_dict
        last_fm_artist_file.close()

    def _read_pandora_results(self, just_read_tags=False):
        pandora_file = open("%s/lists/Pandora/PandoraTagSong.tab" % self.basedir, "r")
        for line in pandora_file:
            pandora_tag = self._read_tag(line)
            if just_read_tags:
                self.only_these_tags.add(pandora_tag)
            elif pandora_tag in self.only_these_tags:
                pandora_dict = self._line_to_dict(line.rstrip().split("\t"))
                try:
                    new_dict = self.ground_truth[pandora_tag]
                    for (songid, pandora_score) in pandora_dict.items():
                        try:
                            lastfm_score = new_dict[songid]
                            new_dict[songid] = max(lastfm_score, pandora_score)
                        except KeyError: # song dict doesn't have this songid
                            new_dict[songid] = pandora_score
                    self.ground_truth[pandora_tag] = new_dict
                except KeyError: # tag doesn't have lastfm info; just use pandora info
                    self.ground_truth[pandora_tag] = pandora_dict
        pandora_file.close()

    """
    I think this is unnecessary, given that I prune when making the X and y matrices....
    def _prune_low_count_tags(self):
        for (tag, song_dict) in self.ground_truth.items():
            if len(song_dict) < self.min_tag_count:
                self._remove_tag(tag, verbosity=0)
    """

    def _remove_uncommon_tags(self):
        feature_tags = set(self.features.keys())
        ground_truth_tags = set(self.ground_truth.keys())
        remove_these = feature_tags.symmetric_difference(ground_truth_tags)
        for tag in remove_these:
            self._remove_tag(tag, verbosity=0)
        if self.verbosity > 1:
            util.info("\tWARNING: Removing these tags not in intersection of features and ground truth: %s." % str(remove_these))
             
    def _compute_sorted_sources(self):
        """
        Preliminary experiments showed that intercept helps a lot, increasing 10precision from 0.05 to 0.06
        """
        self.sorted_sources = dict()
        for (tag, source_dict) in self.features.items():
            source_list = []
            if self.add_intercept_as_first_feature:
                source_list.append("intercept")
            # Need to include main effect if we also have scrobble interaction:
            # http://orgtheory.wordpress.com/2007/10/20/interaction-models/
            if "P" in self.regmodel:
                source_list.append("scrobble")
            # Find the other sources.
            tag_has_nontrivial_sources = False
            for (source, song_dict) in sorted(source_dict.items()): # sort so that features are always in same order
                if len(song_dict) >= self.min_feature_count:
                    source_list.append(source)
                    if "I" in self.regmodel:
                        source_list.append("%s_interaction" % source)
                    tag_has_nontrivial_sources = True
            if tag_has_nontrivial_sources:
                self.sorted_sources[tag] = source_list
            else:
                self._remove_tag(tag, verbosity=0)
    
    def _remove_tags_without_all_sources(self):
        try:
            n_sources = max([len(source_list) for source_list in self.sorted_sources.values()])
        except:
            pdb.set_trace()
        for (tag, source_list) in self.sorted_sources.items():
            if len(source_list) < n_sources:
                self._remove_tag(tag, verbosity=0)

    def _make_X_and_y_matrices(self):
        tag_num = 0
        n_tags = len(self.features)
        for (tag, sources_dict) in self.features.items():
            tag_num += 1
            self._progress("Reading tag %d of %d: %s" % (tag_num, n_tags, tag), newline=True) # rmme: make newline false
            self.song_lists[tag] = self._compute_song_list(sources_dict)
            if len(self.song_lists[tag]) > 0 and tag in self.ground_truth:
                self._cur_tag_X_and_y(tag)
            else:
                self._remove_tag(tag)
                assert False, """We shouldn't get here, because we should already have removed tags not shared by features and ground truth."""


    def _compute_song_list(self, sources_dict, use_entire_song_list=True):
        if use_entire_song_list:
            song_set = self.all_seen_songs
        else:
            song_set = set()
            for (source_name, source_d) in sources_dict.iteritems():
                song_set.update(set(source_d.keys())) # union of songs for all sources; could also do intersection
            song_set = song_set.intersection(self.all_seen_songs)
        song_list = list(song_set)
        random.seed("Another random seed, so that shuffling will be fixed though arbitrary.")
        random.shuffle(song_list)
        return song_list

    def _cur_tag_X_and_y(self, tag, test_random=False, all_ground_truth_binary=True):
        """
        """
        sorted_sources = self.sorted_sources[tag]
        cur_song_list = self.song_lists[tag]
        features_dict = self.features[tag]
        y = rc.matrix(self._dict_to_vec(self.ground_truth[tag], self.song_lists[tag]))
        if util.num_nonzeros(y) < self.min_tag_count:
            self._remove_tag(tag, verbosity=0)
            return
        # Possibly add intercept.
        ncol = 0
        x_vec = []
        if "intercept" in sorted_sources:
            ncol += 1
            x_vec.extend([1.0 for songid in cur_song_list])
        # Add Scrobble counts
        scrobble_vec = numpy.array([self.log_scrobble_counts.get(songid, self.avg_scrobble) for songid in cur_song_list])
        if "scrobble" in sorted_sources:
            ncol += 1
            x_vec.extend(self._standardize(scrobble_vec))
        # Add remaining features.
        for source in sorted_sources:
            if source.endswith("_interaction"):
                main_source = util.remove_trailing_string(source, "_interaction")
                main_vec = self._dict_to_vec(features_dict.get(main_source, None), cur_song_list)
                feature_vec = numpy.multiply(main_vec, scrobble_vec) # pointwise product
            elif source not in ["intercept", "scrobble"]:
                feature_vec = self._dict_to_vec(features_dict.get(source, None), cur_song_list)
            else:
                continue
            x_vec.extend(self._standardize(feature_vec))
            ncol += 1
        try:
            X = rc.matrix(x_vec,ncol=ncol)#dimnames=[[],x_sources])
        except:
            pdb.set_trace()
            # Temporary fix: rmme!
            self._remove_tag(tag)
            return
        if test_random and not self.production_run: # Erase all of the above and do some random numbers for testing.
            n_songs = len(X)
            X = numpy.random.standard_normal((n_songs,ncol))
            if ncol==1:
                y = 3 * X[:,0] + 0.5*numpy.random.standard_normal((1,n_songs))
            else:
                y = 3 * X[:,0] + X[:,1] + 0.5*numpy.random.standard_normal((1,n_songs))
            y = y.transpose()
        if all_ground_truth_binary or self.regtype=="Independent Logistic" or self.regtype=="Hierarchical Logistic":
            # Convert y to 0's and 1's.
            y = 1.0*(numpy.array(y)>0) # multiply by 1.0 to make Float
        self.X[tag] = X
        self.y[tag] = y

    def _standardize(self, vector):
        """
        Makes betas comparable.
        """
        mean = numpy.mean(vector)
        std = numpy.std(vector)
        if std==0: # This happens, e.g., when vector is all 1's, or just a single number.
            return [1.0 for val in vector] # We have to pick something....
        else:
            return map(lambda x: (x-mean)/std, vector)

    def _dict_to_vec(self, d, song_list):
        """
        """
        if d is None:
            # This acts as though we have information we don't. Another approach
            # is to leave these out of the model entirely.
            return numpy.array([0.0 for val in song_list])
        else:
            return numpy.array([d.get(val, 0.0) for val in song_list])

    def _compute_betas(self, remove_tags_when_bad_regression=True):
        subsets_of_sources = self._compute_subsets_of_sources()
        for (source_list, tag_list) in subsets_of_sources.iteritems():
            self._progress("Computing betas for source subset = %s." % source_list, newline=False)
            if self.regmodel=="Random":
                self._random_betas(tag_list)
            else:
                if "Hierarchical" in self.regtype:
                    self._mcmc_betas_same_sources(tag_list)
                elif "Independent" in self.regtype:
                    self._independent_betas_same_sources(tag_list, remove_tags_when_bad_regression)
                elif self.regtype=="Random Guess":
                    self._random_betas(tag_list)
                elif self.regtype in ["Product","Sum","Min","Max","Median"]:
                    self._NA_betas(tag_list)
                else:
                    raise ValueError("Bad regtype = %s." % self.regtype)
        if self.force_betas_nonneg_except_scrobble_or_if_interactions:
            for (tag, source_dict) in self.beta.items():
                for (source, stat_dict) in source_dict.items():
                    if source != "scrobble" and stat_dict["beta"] < 0:
                        # Remove the p-values and other stats; just make beta 0.
                        self.beta[tag][source] = {"beta":0}
                        # TODO: Replace with nonnegative regression.

    def _compute_subsets_of_sources(self):
        subsets = dict()
        for (tag, sorted_source_list) in self.sorted_sources.iteritems():
            subsets.setdefault(str(sorted_source_list), []).append(tag)
        return subsets

    def _mcmc_betas_same_sources(self, tag_list):
        """
        The given tag_list contains tags that all have the same features
        available. Train on the tags in tag_list using only the songs
        in self.only_these_songs, or all available songs if
        self.only_these_songs is None.
        """
        if not self.production_run:
            self.mcmc_reps = 75 # save time
        rc.library("bayesm")
        data = []
        for tag in tag_list:
            data.append(rc.list(X=self.X[tag],y=self.y[tag]))
        rpy.set_default_mode(rpy.NO_CONVERSION) # Turn off conversion so that lm returns Robj.
        data = rc.list(*data)
        if self.regtype in ["Hierarchical Linear", "Hierarchical Mixture"]:
            Data = rc.list(regdata=data)
        elif self.regtype=="Hierarchical Logistic":
            Data = rc.list(lgtdata=data)
        if self.regtype=="Hierarchical Mixture":
            Prior = rc.list(ncomp=self.ncomp)
        Mcmc=rc.list(R=self.mcmc_reps)
        rpy.set_default_mode(rpy.BASIC_CONVERSION)
        try:
            if self.regtype=="Hierarchical Linear":
                output = rc.rhierLinearModel(Data=Data,Mcmc=Mcmc)
            elif self.regtype=="Hierarchical Logistic":
                output = rc.rhierBinLogit(Data=Data,Mcmc=Mcmc)
            elif self.regtype=="Hierarchical Mixture":
                output = rc.rhierLinearMixture(Data=Data,Prior=Prior,Mcmc=Mcmc)
        except:
            #pdb.set_trace()
            self._info_about_r_error(tag_list)
            return
        beta_matrix = output['betadraw'].mean(axis=2) # nregressions x ncoeffs, averaged along third dim
        matrix_index = 0
        for tag in tag_list:
            cur_tag_beta_vec = beta_matrix[matrix_index,:]
            beta_dict_list = [dict([('beta', coeff)]) for coeff in cur_tag_beta_vec]
            self.beta[tag] = dict(zip(self.sorted_sources[tag],beta_dict_list))
            self.stats[tag] = dict() # I'm not currently storing any stats for hierarchical regressions.
            matrix_index += 1

    def _info_about_r_error(self, tag_list):
        """
        Print info when rhier... functions raise a sorting-related error.
        Delete all tags we're working with.
        """
        util.info("\tERROR: Problem with R's sorting thingymajig. %s" % str(sys.exc_info()))
        for tag in tag_list:
            self._remove_tag(tag)

    def _independent_betas_same_sources(self, tag_list, remove_tags_when_bad_regression, n_times_show_summary=3):
        times_showed_summary = 0 # This allows us to print out some summary statistics without producing an overwhelming amount of output.
        SUMMARY_STATS = ["beta", "stderr", "tstat", "pval"]
        for tag in tag_list:
            self._progress("Computing betas for tag %s." % tag, newline=True) # rmme: newline make false
            rpy.set_default_mode(rpy.NO_CONVERSION) # Turn off conversion so that lm returns Robj.
            data = rc.list(y=self.y[tag],X=self.X[tag])
            model = "y~X-1" # Use -1 because X has an intercept already
            if self.regtype=="Independent Linear":
                try:
                    result = rc.lm(model,data=data)
                except:
                    pdb.set_trace()
            elif self.regtype=="Independent Logistic":
                result = rc.glm(model,family=rc.binomial("logit"),data=data)
            rpy.set_default_mode(rpy.BASIC_CONVERSION) # Return to normal conversion mode.
            summary = rc.summary(result,correlation=rc.TRUE)
            self._record_regression_stats(tag, summary)
            beta_dict = dict()
            sorted_sources = self.sorted_sources[tag]
            coeff_matrix = summary["coefficients"]
            for i in range(len(sorted_sources)):
                try:
                    cur_source_dict = dict(zip(SUMMARY_STATS,coeff_matrix[i,:]))
                except IndexError:
                    util.info("\tWARNING: Regression for %s didn't end up using all variables." % tag)
                    if remove_tags_when_bad_regression:
                        self._remove_tag(tag)
                        break # break from for-loop over sorted_sources; we don't continue out of the per-tag for loop until later when we check if tag is in self.features....
                    continue
                try:
                    cur_source_dict["-log10(pval)"] = -log(cur_source_dict["pval"], 10)
                except OverflowError:
                    pass
                beta_dict[sorted_sources[i]] = cur_source_dict
            if tag not in self.features: # We've removed this tag a few lines above, so skip it.
                continue
            self.beta[tag] = beta_dict
            if times_showed_summary < n_times_show_summary:
                self._print_regression_summary(tag, summary)
                times_showed_summary += 1
            #predictions = rc.predict(result)
            #predictions = [predictions['%d' % i] for i in range(1,len(predictions)+1)]

    def _random_betas(self, tag_list):
        """
        Betas should be just 0.
        """
        for tag in tag_list:
            beta_dict = dict()
            for source in self.sorted_sources[tag]:
                beta_dict[source] = dict([("beta",0)])
            self.beta[tag] = beta_dict
            self.stats[tag] = dict()

    def _NA_betas(self, tag_list):
        """
        Just fill in the beta_dict so that we can use it to iterate through tags later.
        """
        for tag in tag_list:
            beta_dict = dict()
            for source in self.sorted_sources[tag]:
                beta_dict[source] = dict([("beta","NA because regtype is %s." % self.regtype)])
            self.beta[tag] = beta_dict
            self.stats[tag] = dict()

    def _record_regression_stats(self, tag, summary):
        self.stats[tag] = dict()
        if self.regtype=="Independent Linear":
            self.stats[tag][r"$R^2$"] = summary['r.squared']
            self.stats[tag][r"$R^2_\text{adj}$"] = summary["adj.r.squared"]
            self.stats[tag][r"$\widehat{\sigma}$"] = summary["sigma"]
            try:
                self.stats[tag][r"$-log_10(\text{p-val})"] = -log(1-rc.pf(summary['fstatistic']['value'],summary['fstatistic']['numdf'],summary['fstatistic']['dendf']), 10)  # For why this formula works, see, e.g., http://cran.r-project.org/doc/contrib/Faraway-PRA.pdf (page 30)
            except OverflowError:
                pass # p-value is really really small!
        elif self.regtype=="Independent Logistic":
            self.stats[tag]["Deviance"] = summary["deviance"]  # The deviance is twice the log-likelihood ratio statistic
            self.stats[tag]["Null deviance"] = summary["null.deviance"]
            self.stats[tag]["AIC"] = summary["aic"]

    def _print_regression_summary(self, tag, summary):
        self._progress("Cur tag = %s." % tag)
        if self.regtype=="Independent Linear":
            self._progress("R^2 = %.3f, R^2-adjusted = %.3f, estim err sigma = %.2f." % (summary['r.squared'], summary["adj.r.squared"], summary["sigma"]))
        self._progress("Feature correlation matrix:\n%s" % str(numpy.corrcoef(numpy.transpose(self.X[tag]))))
        self._progress("Beta correlation matrix:\n%s" % str(summary['correlation']))

    def _compute_yhat(self):
        self._progress("Computing yhat.")
        for (tag, X_test) in self.X.iteritems():
            X_test_sources = self.sorted_sources[tag]
            n_songs = len(X_test)
            yhat = numpy.zeros((1,n_songs))
            if self.regtype in ["Product","Sum","Min","Max","Median"]:
                if self.add_intercept_as_first_feature:
                    assert X_test[0][0]==1, "Failed quick-and-dirty test for intercept."
                    X_test = X_test[:,1:] # eliminate intercept
                feature_probs = 1/(1+numpy.exp(-X_test)) # Convert positive and negative scores to "probabilities".
                for i in range(numpy.size(yhat,axis=1)):
                    yhat[0,i] = self._combine_probs(feature_probs[i,:])
            else:
                for (source, stats_dict) in self.beta[tag].items():
                    try:
                        index_in_X_test = X_test_sources.index(source)
                        yhat += stats_dict['beta'] * X_test[:,index_in_X_test]
                    except ValueError:
                        continue
                if self.regtype=="Independent Logistic" or self.regtype=="Hierarchical Logistic":
                    yhat = 1/(1+numpy.exp(-yhat))
            self.yhat[tag] = yhat

    def _combine_probs(self, feature_probs):
        """
        feature_probs is an array of "probabilities" to be combined.
        """
        if self.regtype=="Product":
            return numpy.exp(numpy.sum(numpy.log(feature_probs)))
        elif self.regtype=="Sum":
            return numpy.sum(feature_probs)
        elif self.regtype=="Max":
            return max(feature_probs)
        elif self.regtype=="Min":
            return min(feature_probs)
        elif self.regtype=="Median":
            return numpy.median(feature_probs)
        else:
            raise ValueError("You shouldn't be calling this function with regtype=%s." % self.regtype)

    def _performance_stats(self, remove_tags_dont_work=True):
        self._progress("Computing performance stats.")
        for (tag, true_y) in self.y.items():
            if tag not in self.stats:
                pdb.set_trace()
            yhat = self.yhat[tag].transpose() # make n_songs x 1
            self.stats[tag]["Num Songs"] = len(yhat)
            # SSE
            self.stats[tag][r'$\text{SSE} / n$'] = numpy.sum(numpy.power((true_y-yhat),2)) / len(true_y)
            # precision, recall, etc.
            sorted_yhat = sorted([(yhat[i,0], i) for i in range(len(yhat))], reverse=True)
            graded = [self._in_ground_truth(true_y[i,0]) for (yhat_val, i) in sorted_yhat]
            try:
                self.stats[tag]["Baseline"] = self._random_precision(graded)
                self.stats[tag]["AUC"] = self._areaUnderCurve(graded)
                self.stats[tag]["MAP"] = self._avgPrecision(graded)
                self.stats[tag]["R-Prec"] = self._rPrecision(graded)
                self.stats[tag]["10-Prec"] = self._tenPrecision(graded)
                baseline = self.stats[tag]["Baseline"]
                if baseline > 0:
                    self.stats[tag]["MAP/Baseline"] = self.stats[tag]["MAP"] / baseline
                    self.stats[tag]["R-Prec/Baseline"] = self.stats[tag]["R-Prec"] / baseline
                    self.stats[tag]["10-Prec/Baseline"] = self.stats[tag]["10-Prec"] / baseline
            except ValueError:
                util.info("WARNING: TP==0 or FP==0 for tag = %s." % tag)
                if remove_tags_dont_work:
                    self._remove_tag(tag)
                    continue
            # Record best and worst songs.
            song_list = list(self.song_lists[tag])
            self.best_worst_songs[tag] = dict()
            index_best_song = sorted_yhat[0][1]
            self.best_worst_songs[tag]["Best Song"] = (self.songid_to_song[song_list[index_best_song]], 1 if true_y[index_best_song,0] else 0)
            index_worst_song = sorted_yhat[-1][1]
            self.best_worst_songs[tag]["Worst Song"] = (self.songid_to_song[song_list[index_worst_song]], 1 if true_y[index_worst_song,0] else 0)

    def _in_ground_truth(self, y_val):
        return y_val > 0

    def _point_precision(self, retrieved_docs):
	"""
	Return the precision when we retrieve the documents whose correctness is
	given by a True or False value in the list retrieved_docs.
	"""
	# Sum of True is 1, False is 0.
        try:
            return sum(retrieved_docs) / len(retrieved_docs)
        except ZeroDivisionError:
            raise ValueError("No retrieved docs.")

    def _random_precision(self, graded):
        self._progress("Computing baseline precision.", newline=False)
        try:
            return sum(graded) / len(graded)
        except ZeroDivisionError:
            raise ValueError("No songs for this tag.")        
	    
    def _avgPrecision(self, graded):
	"""
	Given the list, graded, of True/False values in the order in which we
	ranked the documents, return our average precision.
	"""
        self._progress("Computing avg precision.", newline=False)
        TP = 0
        P = 0 # total num called positive
        running_sum_for_avg_precision = 0
        for item in graded:
            P += 1
            if item:
                TP += 1
                running_sum_for_avg_precision += TP / P
        return running_sum_for_avg_precision / TP
            
    def _rPrecision(self, graded):
	"""
	Given the list, graded, of True/False values in the order in which we
	ranked the documents, return R-precision.
	"""	
        self._progress("Computing R-precision.", newline=False)        
	R = sum(graded) # Num relevant.
	return self._point_precision(graded[:R])
    
    def _tenPrecision(self, graded):
	"""
	Given the list, graded, of True/False values in the order in which we
	ranked the documents, return the 10-precision.
	"""
        self._progress("Computing 10-precision.", newline=False)        
	n_return = min(len(graded), 10)
	return self._point_precision(graded[:n_return])
    
    def _areaUnderCurve(self, graded):
	"""
	Given the list, graded, of True/False values in the order in which we
	ranked the documents, return the AUC.
	"""
        self._progress("Computing AUC.", newline=False)
	sum = 0
	TP = 0
	FP = 0
	for boolVal in graded:
	    if boolVal:
		TP += 1
	    else:
		FP += 1
		sum += TP
	if TP == 0 or FP == 0:
	    raise ValueError("AUC undefined.")
        return sum / (TP * FP)

    def _write_final(self, outfile):
        self._progress("Writing interpolated results to %s." % outfile)
        if self.overwrite_final_tab_file:
            file = open(outfile, "w")
        else:
            file = open(outfile, "a")
        for (tag, true_y) in self.y.iteritems():
            song_score_pairs = []
            interpolated = numpy.maximum(true_y, self.yhat[tag].transpose())
            song_list = list(self.song_lists[tag])
            for i in range(len(interpolated)):
                song_score_pairs.append((song_list[i], interpolated[i,0]))
            file.write(tag)
            for (song, score) in sorted(song_score_pairs, key=lambda (song, score): score, reverse=True):
                file.write("\t%d\t%f" % (song, score))
            file.write("\n")
        file.close()

def main(a_few_tags=False):
    if a_few_tags:
        A_FEW_TAGS = ['contemporary gospel']#,'yellow','a blues song form','country influences',"post rock","1970s"]
        Combiner(only_these_tags=set(A_FEW_TAGS), production_run=True).fill_in_zeros()
    else:
        # We run out of memory trying to do all tags at once, so just do 2000 at a time.
        N_TAGS_PER_ROUND = 2000
        PRODUCTION_RUN = True
        combiner = Combiner(production_run=PRODUCTION_RUN)
        all_tags = list(combiner.only_these_tags)
        tag_groups = util.partition(all_tags, N_TAGS_PER_ROUND)
        overwrite_final_tab_file = True
        for group in tag_groups:
            group_combiner = Combiner(production_run=PRODUCTION_RUN, only_these_tags=set(group), overwrite_final_tab_file=overwrite_final_tab_file)
            overwrite_final_tab_file = False # From now on, we'll just append to the current one.
            group_combiner.fill_in_zeros()

if __name__ == "__main__":
    main()

