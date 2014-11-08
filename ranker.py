# Ranker class

from __future__ import division
import pdb
import sys

__author__ = "Brian Tomasik"
__date__ = "Apr. 2009"

class Ranker(object):
    """
    A Ranker reads the "final.tab" file of combined scores and uses them
    to answer queries with ranked lists of songs.
    """
 
    def __init__(self, combined_file="final.tab", cutoff=.05, verbosity=1):
        """
        Read the combined-scores .tab file into a dictionary by tag.
        Ignore songs with scores less than cutoff, to save memory.
        Also create a forward index for songs.
        """
        self.tags = dict()
        self.verbosity = verbosity
        file = open(combined_file, "r")
        for line in file:
            line_list = line.rstrip().split("\t")
            tag = line_list[0]
            if self.verbosity > 0:
                sys.stderr.write("Adding tag %s.\r" % tag)
            pairs_list = self._list_to_pairs(line_list[1:], cutoff)
            self.tags[tag] = pairs_list
        file.close()
        if self.verbosity > 0:
            sys.stderr.write("\n")
        self.songs = self._compute_forward_index(self.tags)
        if self.verbosity > 0:
            sys.stderr.write("\n")

    def _list_to_pairs(self, flat_list, cutoff):
        """
        Given a list [song, score, song, score, ...], convert it to a list
        of pairs [(song, score), (song, score), ...]. Change the scores
        from strings to floats. Keep only pairs with score >= cutoff.
        """
        pairs_list = []
        for i in range(len(flat_list)):
            if i % 2 == 0:
                cur_score = float(flat_list[i+1])
                if cur_score >= cutoff:
                    pairs_list.append((int(flat_list[i]), cur_score))
        return pairs_list

    def _compute_forward_index(self, inverted_index):
        """
        Given an inverted_index, compute the corresponding forward
        index. The inverted index is a dictionary mapping to a list
        of sorted (song, score) pairs, while the forward index will be a
        dictionary mapping to dictionaries over tags.
        """
        forward_index = dict()
        for (tag, song_list) in inverted_index.items():
            if self.verbosity > 0:
                sys.stderr.write("Computing forward index with inverted index for tag %s.\r" % tag)
            for (song, score) in song_list:
                cur_dict = forward_index.get(song, dict())
                cur_dict[tag] = score
                forward_index[song] = cur_dict
        return forward_index

    def answer_query(self, tag_dict, n_songs=50):
	"""
        Given a query dictionary over tags and a number of songs, return
        a list of that many songs as (songid, rank_score, tag_dict) triples.
	"""
        song_scores = dict()
        for (tag, weight) in sorted( \
            tag_dict.items(), key=lambda(tag, weight): weight, reverse=True):
            try:
                songs_for_tag = self.tags[tag]
            except KeyError:
                continue
            for (songid, score) in songs_for_tag:
                song_scores[songid] = song_scores.get(songid, 0.0) + \
                    weight * score
        ranked_list = sorted(song_scores.items(), \
                                 key=lambda(songid, score): score, \
                                 reverse=True)
        ranked_list = ranked_list[:n_songs]
        tag_adder = lambda (songid, score): (songid, score, self.songs[songid])
        return map(tag_adder, ranked_list)

def main():
    ranker = Ranker("final.tab")
    query = {'rap':0.4, 'classical':0.6}
    print ranker.answer_query(query)

if __name__ == "__main__":
    main()
