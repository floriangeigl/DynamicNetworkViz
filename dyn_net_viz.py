from __future__ import division
from sys import platform as _platform
import matplotlib

if _platform == "linux" or _platform == "linux2":
    matplotlib.use('Agg')
import matplotlib.pylab as plt
import matplotlib.cm as colormap
from matplotlib.colors import ColorConverter as color_converter
from graph_tool.all import *
import numpy as np
import pandas as pd
import Image
import ImageDraw
import ImageFont
import datetime
import shutil
import copy
import random
import operator
import math
import subprocess
import os
import printing
from collections import defaultdict

__author__ = 'Florian'


def create_folder_structure(filename):
    d = os.path.dirname(filename)
    if not os.path.exists(d):
        os.makedirs(d)


def print_f(*args, **kwargs):
    if 'class_name' not in kwargs:
        kwargs.update({'class_name': 'gt_tools'})
    printing.print_f(*args, **kwargs)


class graph_viz():
    def __init__(self, dataframe, network, filename='output/network_evolution.png', verbose=1, df_iteration_key='iteration', df_vertex_key='vertex', df_opinion_key=None,
                 df_state_key=None, df_edges_key=None, plot_each=1, ips=1, output_size=(1920, 1080), bg_color='white', fraction_groups=None, smoothing=1, rate=30, cmap=None,
                 inactive_fraction_f=lambda x: x == {-1}, inactive_value_f=lambda x: x <= 0, deactivated_color_nodes=None, mu=0.0, mark_new_active_nodes=False,
                 pause_after_iteration=0, largest_component_only=False, edge_blending=False, keep_inactive_nodes=True, max_node_alpha=0.9):
        assert isinstance(dataframe, pd.DataFrame)
        self.df = dataframe
        self.df_iteration_key = df_iteration_key
        self.df_vertex_key = df_vertex_key
        self.df_op_key = df_opinion_key
        self.df_edges_key = df_edges_key
        self.draw_fractions = self.df_op_key is not None
        self.df_state_key = df_state_key
        self.lc_only = largest_component_only
        self.edge_blending = edge_blending
        self.keep_inactive_nodes = keep_inactive_nodes
        if any(isinstance(i, int) for i in self.df_vertex_key):
            self.print_f('convert vertex column to vertex instances')
            self.df[self.df_vertex_key] = self.df[self.df_vertex_key].apply(func=lambda x: network.vertex(x))

        self.categories = set.union(*list(self.df[self.df_op_key])) if self.df_op_key is not None else None
        self.network = network
        assert isinstance(self.network, Graph)
        self.output_filenum = 0

        filename = filename if filename.endswith('.png') else filename + '.png'
        filename = filename.rsplit('/', 1)
        if len(filename) == 1:
            filename = ['.', filename[0]]
        filename[1] = str('_' + filename[1])
        filename = '/'.join(filename)
        self.filename = filename
        splited_filename = self.filename.rsplit('/', 1)
        self.filename_folder = splited_filename[0]
        self.filename_basename = splited_filename[-1]
        self.tmp_folder_name = 'graph_animator_tmp/'
        self.pause_after_iteration = pause_after_iteration
        self.edges_filename = self.filename_folder + '/' + self.tmp_folder_name + 'edges_' + self.filename_basename
        self.mu = mu
        if not os.path.isdir(self.filename_folder + '/' + self.tmp_folder_name):
            try:
                create_folder_structure(self.filename_folder + '/' + self.tmp_folder_name)
            except:
                self.print_f('Could not create tmp-folder:', self.filename_folder + '/' + self.tmp_folder_name)
                raise Exception
        self.verbose = verbose
        self.plot_each = plot_each
        self.mark_new_active_nodes = mark_new_active_nodes
        self.ips = ips
        self.output_size = output_size
        if isinstance(self.output_size, (int, float)):
            self.output_size = (self.output_size, self.output_size)
        self.color_converter = color_converter()
        self.bg_color = self.color_converter.to_rgba(bg_color)
        self.fraction_groups = fraction_groups
        self.smoothing = smoothing
        self.rate = rate
        self.generate_files = None
        self.pos = None
        self.pos_abs = None
        self.cmap = cmap
        self.inactive_fraction_f = inactive_fraction_f
        self.inactive_value_f = inactive_value_f
        self.active_nodes = None
        self.active_edges = None
        self.max_node_alpha = max_node_alpha
        if self.cmap is None:
            def_cmap = 'gist_rainbow'
            self.print_f('using default cmap:', def_cmap, verbose=2)
            self.cmap = colormap.get_cmap(def_cmap)
        elif isinstance(self.cmap, str):
            self.cmap = colormap.get_cmap(self.cmap)
        self.deactivated_color_nodes = [0.179, 0.179, 0.179, 0.05] if deactivated_color_nodes is None else deactivated_color_nodes
        efilt = self.network.new_edge_property('bool')
        for e in self.network.edges():
            efilt[e] = False
        self.network.ep['no_edges_filt'] = efilt
        self.first_iteration = True

    def generate_filename(self, filenum):
        return self.filename_folder + '/' + self.tmp_folder_name + str(int(filenum)).rjust(6, '0') + self.filename_basename

    def get_color_mapping(self, categories=None, groups=None, cmap=None):
        self.print_f('get color mapping', verbose=2)
        if cmap is None:
            cmap = colormap.get_cmap('gist_rainbow')
        if groups and categories:
            try:
                g_cat = set.union(*[groups[i] for i in categories])
                g_cat_map = {i: idx for idx, i in enumerate(g_cat)}
                num_g_cat = len(g_cat)
                color_mapping = {i: g_cat_map[random.sample(groups[i], 1)[0]] / num_g_cat for i in sorted(categories)}
            except:
                self.print_f('Error in getting categories color mapping.', traceback.print_exc(), verbose=1)
                return self.get_color_mapping(categories, cmap=cmap)
        elif categories:
            num_categories = len(categories)
            color_mapping = {i: idx / num_categories for idx, i in enumerate(sorted(categories))}
        else:
            return cmap
        result = {key: (cmap(val), val) for key, val in color_mapping.iteritems()}
        result.update({-1: (self.deactivated_color_nodes, -1)})
        return result

    def print_f(self, *args, **kwargs):
        kwargs.update({'class_name': 'GraphAnimator'})
        if 'verbose' not in kwargs or kwargs['verbose'] <= self.verbose:
            print_f(*args, **kwargs)

    def calc_absolute_positions(self, init_pos=None, network=None, mu=None, **kwargs):
        if network is None:
            network = self.network
        if mu is None:
            mu = self.mu
        if init_pos is not None:
            pos = copy.copy(init_pos)
        else:
            pos = sfdp_layout(network, mu=mu, **kwargs)
        pos_ar = pos.get_2d_array((0, 1)).T
        if self.lc_only:
            lc = label_largest_component(network)
            network = GraphView(network, vfilt=lc)
        if isinstance(network, GraphView):
            pos_ar_stat = pos_ar[list(map(int, network.vertices()))]
        else:
            pos_ar_stat = pos_ar
        max_v = pos_ar_stat.max(axis=0)
        min_v = pos_ar_stat.min(axis=0)
        max_v -= min_v
        spacing = 0.15 if network.num_vertices() > 10 else 0.3
        shift_mult = np.array(self.output_size) * (1 - spacing)
        shift_add = np.array(self.output_size) * (spacing / 2)
        max_v += (np.array([1, 1]) * (max_v == 0))
        print 'max:', max_v
        mult = (1 / max_v) * shift_mult
        pos.set_2d_array(((pos_ar - min_v) * mult + shift_add).T)
        print pos.get_2d_array((0,1)).T
        return pos

    def calc_grouped_sfdp_layout(self, network=None, groups_vp='groups', pos=None, mu=None, **kwargs):
        if network is None:
            network = self.network
        if mu is None:
            mu = self.mu
        orig_groups_map = network.vp[groups_vp] if isinstance(groups_vp, str) else groups_vp
        e_weights = network.new_edge_property('float')
        for e in network.edges():
            src_g, dest_g = orig_groups_map[e.source()], orig_groups_map[e.target()]
            try:
                e_weights[e] = len(src_g & dest_g) / len(src_g | dest_g)
            except ZeroDivisionError:
                e_weights[e] = 0
        groups_map = network.new_vertex_property('int')
        for v in network.vertices():
            v_orig_groups = orig_groups_map[v]
            if len(v_orig_groups) > 0:
                groups_map[v] = random.sample(v_orig_groups, 1)[0]
            else:
                groups_map[v] = -1
        return sfdp_layout(network, pos=pos, groups=groups_map, eweight=e_weights, mu=mu, **kwargs)

    def plot_network_evolution(self, dynamic_pos=False, infer_size_from_fraction=True, delete_pictures=True, label_pictures=True):
        self.output_filenum = 0
        tmp_smoothing = self.ips * self.smoothing
        smoothing = self.smoothing
        fps = self.ips
        while tmp_smoothing > self.rate:
            smoothing -= 1
            tmp_smoothing = fps * smoothing
        smoothing = max(1, smoothing)
        fps *= smoothing
        init_pause_time = 3 * fps / smoothing

        if init_pause_time == 0:
            init_pause_time = 3
        init_pause_time = int(math.ceil(init_pause_time))
        self.print_f('Framerate:', fps, verbose=2)
        self.print_f('Iterations per second:', fps / smoothing, verbose=2)
        self.print_f('Smoothing:', smoothing, verbose=2)
        self.print_f('Init pause:', init_pause_time, verbose=2)

        # get colors
        color_mapping = self.get_color_mapping(self.categories, self.fraction_groups, self.cmap)

        # get positions
        if not dynamic_pos:
            self.print_f('calc graph layout', verbose=1)
            try:
                self.pos = self.calc_grouped_sfdp_layout(network=self.network, groups_vp='groups')
            except KeyError:
                self.pos = sfdp_layout(self.network, mu=self.mu)
            # calc absolute positions
            self.pos_abs = self.calc_absolute_positions(self.pos, network=self.network)

        # PLOT
        total_iterations = self.df[self.df_iteration_key].max() - self.df[self.df_iteration_key].min()

        self.print_f('iterations:', total_iterations, verbose=1)
        if self.draw_fractions:
            self.print_f('draw fractions')
            self.network.vp[self.df_op_key] = self.network.new_vertex_property('object')
            fractions_vp = self.network.vp[self.df_op_key]
            for v in self.network.vertices():
                fractions_vp[v] = set()
        else:
            fractions_vp = None

        vertex_state = self.network.new_vertex_property('float')
        if self.df_state_key is not None:
            self.network.vp[self.df_state_key] = vertex_state

        try:
            _ = self.network.vp['NodeId']
        except KeyError:
            mapping = self.network.new_vertex_property('int')
            for v in self.network.vertices():
                mapping[v] = int(v)
            self.network.vp['NodeId'] = mapping

        self.df[self.df_iteration_key] = self.df[self.df_iteration_key].astype(int)
        grouped_by_iteration = self.df.groupby(self.df_iteration_key)
        self.print_f('Resulting video will be ~', int(total_iterations / self.plot_each * smoothing / fps) + (init_pause_time * 2 / fps * smoothing) + int(
            total_iterations / self.plot_each * self.pause_after_iteration), 'seconds long')

        self.print_f('Iterations with changes:', ', '.join([str(i) for i, j in grouped_by_iteration]), verbose=2)
        self.network.ep['active_edges'] = self.network.new_edge_property('float')
        if self.draw_fractions:
            self.network.vp['node_color'] = self.network.new_vertex_property('object')
            self.network.vp['node_fractions'] = self.network.new_vertex_property('vector<float>')
        else:
            self.network.vp['node_color'] = self.network.new_vertex_property('vector<float>')
        self.print_f('calc init positions')
        if dynamic_pos or self.pos is None:
            self.pos = sfdp_layout(self.network, mu=self.mu)
            self.print_f('calc init abs-positions')
            self.pos_abs = self.calc_absolute_positions(self.pos, network=self.network)
        self.first_iteration = True

        start = datetime.datetime.now()
        last_iteration = self.df[self.df_iteration_key].min() - 1
        draw_edges = False
        just_copy = True
        last_progress_perc = -1
        iteration_idx = -1
        num_just_copied = 0
        self.generate_files = defaultdict(list)
        active_edges = None if self.df_edges_key is None else set()
        for iteration, data in grouped_by_iteration:
            for one_iteration in range(last_iteration + 1, iteration + 1):
                iteration_idx += 1
                last_iteration = one_iteration
                self.print_f('iteration:', one_iteration, verbose=2)
                if one_iteration == iteration:
                    for idx, row in filter(lambda lx: isinstance(lx[1][self.df_vertex_key], Vertex), data.iterrows()):
                        vertex = row[self.df_vertex_key]
                        if self.draw_fractions:
                            old_f_vp = fractions_vp[vertex]
                            new_f_vp = row[self.df_op_key]
                            if not draw_edges:
                                len_old, len_new = len(old_f_vp), len(new_f_vp)
                                if len_old != len_new and (len_old == 0 or len_new == 0):
                                    draw_edges = True
                            if just_copy:
                                if old_f_vp != new_f_vp:
                                    just_copy = False
                            fractions_vp[vertex] = new_f_vp
                        else:
                            old_size = vertex_state[vertex]
                            new_size = row[self.df_state_key]
                            if not draw_edges and old_size != new_size and (old_size == 0 or new_size == 0):
                                draw_edges = True
                            if just_copy:
                                if old_size != new_size:
                                    just_copy = False
                            vertex_state[vertex] = new_size
                        if active_edges is not None:
                            new_active_edges = row[self.df_edges_key]
                            if hasattr(new_active_edges, '__iter__'):
                                if not isinstance(new_active_edges, set):
                                    new_active_edges = set(new_active_edges)
                                active_edges.update(new_active_edges)
                            else:
                                active_edges.add(new_active_edges)
                        self.print_f(one_iteration, vertex, 'has', fractions_vp[vertex] if self.draw_fractions else (vertex_state[vertex] if vertex_state is not None else ''), verbose=2)
                if iteration_idx % self.plot_each == 0 or iteration_idx == 0 or iteration_idx == total_iterations:
                    current_perc = int(iteration_idx / total_iterations * 100)
                    if just_copy:
                        num_just_copied += 1
                    if iteration_idx - num_just_copied > 0:
                        avg_time = (datetime.datetime.now() - start).total_seconds() / (iteration_idx - num_just_copied)
                        est_time = datetime.timedelta(seconds=int(avg_time * (total_iterations - iteration_idx)))
                    else:
                        est_time = '-'
                    if self.verbose >= 2 or current_perc > last_progress_perc:
                        last_progress_perc = current_perc
                        ext = ' | just copy' if just_copy else (' | draw edges' if draw_edges else '')
                        self.print_f('plot network evolution iteration:', one_iteration, '(' + str(current_perc) + '%)', 'est remain:', est_time, ext, verbose=1)
                    if iteration_idx == 0 or iteration_idx == total_iterations:
                        for i in xrange(init_pause_time):
                            if i == 1:
                                self.print_f('init' if iteration_idx == 0 else 'exit', 'phase', verbose=2)
                            self.generate_files[one_iteration].extend(
                                self.__draw_graph_animation_pic(color_mapping, vertex_state_map=vertex_state, fraction_map=fractions_vp, draw_edges=draw_edges, just_copy_last=(i != 0 or just_copy),
                                                                smoothing=smoothing, dynamic_pos=dynamic_pos, infer_size_from_fraction=infer_size_from_fraction,
                                                                edge_map=active_edges))
                    else:
                        self.generate_files[one_iteration].extend(
                            self.__draw_graph_animation_pic(color_mapping, vertex_state_map=vertex_state, fraction_map=fractions_vp, draw_edges=draw_edges, smoothing=smoothing,
                                                            just_copy_last=just_copy, dynamic_pos=dynamic_pos, infer_size_from_fraction=infer_size_from_fraction,
                                                            edge_map=active_edges))
                    if self.pause_after_iteration > 0:
                        last_img_fn = self.generate_files[one_iteration][-1]
                        for pause_pic in range(self.pause_after_iteration*fps):
                            pause_pic_fn = self.generate_filename(self.output_filenum)
                            shutil.copy(last_img_fn, pause_pic_fn)
                            self.generate_files[one_iteration].append(pause_pic_fn)
                            self.output_filenum += 1
                    # print iteration, ':', self.generate_files[one_iteration][0], '-', self.generate_files[one_iteration][-1]
                    draw_edges = False
                    just_copy = True
        if label_pictures:
            self.label_output()
        self.create_video(fps, delete_pictures)
        return self.df, self.network

    def __draw_graph_animation_pic(self, color_map=colormap.get_cmap('gist_rainbow'), vertex_state_map=None, fraction_map=None, draw_edges=True, just_copy_last=False, smoothing=1,
                                   dynamic_pos=False, infer_size_from_fraction=True, edge_map=None):
        generated_files = []
        if just_copy_last:
            min_filenum = self.output_filenum
            if min_filenum == 0:
                empty_img = Image.new("RGBA", self.output_size, tuple([int(i*255) for i in self.bg_color]))
                empty_img.save(self.generate_filename(min_filenum), 'PNG')
                orig_filename = self.generate_filename(min_filenum)
            else:
                orig_filename = self.generate_filename(min_filenum - 1)
            for smoothing_step in range(smoothing):
                filename = self.generate_filename(self.output_filenum)
                if orig_filename != filename:
                    shutil.copy(orig_filename, filename)
                generated_files.append(filename)
                self.output_filenum += 1
            self.print_f('Copy file:', orig_filename, ' X ', smoothing, verbose=2)
            return generated_files
        default_edge_alpha = (1 / np.log2(self.network.num_edges())) if self.network.num_edges() > 100 else 0.9
        default_edge_color = [0.3, 0.3, 0.3, default_edge_alpha]
        deactivated_edge_alpha = (1 / self.network.num_edges()) if self.network.num_edges() > 0 else 0
        deactivated_edge_color = [0.3, 0.3, 0.3, deactivated_edge_alpha]

        min_vertex_size_shrinking_factor = 2
        v_state = self.network.new_vertex_property('float')
        if not infer_size_from_fraction and vertex_state_map is None or (infer_size_from_fraction and fraction_map is None):
            infer_size_from_fraction = False
            if vertex_state_map is None:
                v_state.a = [1] * self.network.num_vertices()
            else:
                v_state = copy.copy(vertex_state_map)

        colors = self.network.vp['node_color']
        interpolated_size = self.network.new_vertex_property('float')
        active_nodes = self.network.new_vertex_property('bool')
        active_edges = self.network.new_edge_property('bool')
        edge_color = self.network.new_edge_property('vector<float>')
        if self.network.num_edges() > 0:
            edge_color.set_2d_array(np.array([np.array(default_edge_color) for i in range(self.network.num_edges())]).T)
            active_edges.a = np.array([1] * len(active_edges.a))

        nodes_graph = GraphView(self.network, efilt=self.network.ep['no_edges_filt'])
        edges_graph = None
        if draw_edges or dynamic_pos:
            edges_graph = self.network

        if self.draw_fractions:
            if self.first_iteration:
                last_fraction_map = copy.copy(fraction_map)
                self.network.vp['last_fraction_map'] = last_fraction_map
            else:
                last_fraction_map = self.network.vp['last_fraction_map']

            current_fraction_map = nodes_graph.new_vertex_property('object')
            vanish_fraction = nodes_graph.new_vertex_property('object')
            emerge_fraction = nodes_graph.new_vertex_property('object')
            vanish_fraction_reduce = nodes_graph.new_vertex_property('float')
            emerge_fraction_increase = nodes_graph.new_vertex_property('float')
            stay_fraction_change = nodes_graph.new_vertex_property('float')
            interpolated_fraction_vals = nodes_graph.new_vertex_property('vector<float>')
            fraction_mods = nodes_graph.new_vertex_property('vector<int>')
            for v in self.network.vertices():
                new_frac = fraction_map[v]
                last_frac = last_fraction_map[v]
                new_frac_len = len(new_frac)
                last_frac_len = len(last_frac)
                if last_frac_len == 0:
                    last_frac = {-1}
                    # last_frac_len = 1
                if new_frac_len == 0:
                    new_frac = {-1}
                    new_frac_len = 1
                if infer_size_from_fraction:
                    v_state[v] = new_frac_len
                current_frac = last_frac | new_frac
                current_fraction_map[v] = current_frac
                vanish = last_frac - new_frac
                vanish_fraction[v] = vanish
                emerge = new_frac - last_frac
                emerge_fraction[v] = emerge
                old_slice_size = 1 / len(last_frac) if len(last_frac) > 0 else 1
                new_slice_size = 1 / len(new_frac) if len(new_frac) > 0 else 1
                vanish_fraction_reduce[v] = -old_slice_size / smoothing
                emerge_fraction_increase[v] = new_slice_size / smoothing
                stay_fraction_change[v] = (new_slice_size - old_slice_size) / smoothing
                colors[v] = zip(*sorted([color_map[i] for i in current_frac], key=operator.itemgetter(1)))[0]
                colors[v] = [(i[0], i[1], i[2], min(i[3], self.max_node_alpha)) for i in colors[v]]
                tmp_current_fraction_values = []
                sorted_fractions = sorted(current_frac, key=lambda x: color_map[x][1])
                tmp_fraction_mod = []
                for i in sorted_fractions:
                    if i in emerge:
                        tmp_current_fraction_values.append(0)
                        tmp_fraction_mod.append(1)
                    else:
                        if i in vanish:
                            tmp_fraction_mod.append(-1)
                        else:
                            tmp_fraction_mod.append(0)
                        tmp_current_fraction_values.append(old_slice_size)
                fraction_mods[v] = tmp_fraction_mod
                interpolated_fraction_vals[v] = tmp_current_fraction_values
                if self.inactive_fraction_f(new_frac):
                    if edges_graph is not None:
                        for e in v.all_edges():
                            edge_color[e] = [0, 0, 0, 0] if dynamic_pos else deactivated_edge_color
                            active_edges[e] = False
                else:
                    active_nodes[v] = True
                    if edges_graph is not None and edge_map is not None:
                        for e in filter(lambda le: le not in edge_map, v.all_edges()):
                            edge_color[e] = [0, 0, 0, 0] if dynamic_pos else deactivated_edge_color
                            active_edges[e] = False
        else:
            if isinstance(color_map, dict):
                v_color_vals = v_state
            else:
                v_color_vals = prop_to_size(v_state, mi=0, ma=1, power=1)
            for v in self.network.vertices():
                val = v_state[v]
                inactive = self.inactive_value_f(val)
                colors[v] = color_map(v_color_vals[v]) if not inactive else (self.deactivated_color_nodes if not dynamic_pos else (self.deactivated_color_nodes[:3] + [0]))
                colors[v].a[-1] = min(colors[v].a[-1], self.max_node_alpha)
                if inactive:
                    if edges_graph is not None:
                        for e in v.all_edges():
                            edge_color[e] = [0, 0, 0, 0] if dynamic_pos else deactivated_edge_color
                            active_edges[e] = False
                else:
                    active_nodes[v] = True
                    if edges_graph is not None and edge_map is not None:
                        for e in filter(lambda le: le not in edge_map, v.all_edges()):
                            edge_color[e] = [0, 0, 0, 0] if dynamic_pos else deactivated_edge_color
                            active_edges[e] = False

        if self.first_iteration:
            last_edge_color = edge_color
            last_node_color = self.network.new_vertex_property('vector<float>')
        else:
            last_edge_color = self.network.ep['edge_color']
            last_node_color = self.network.vp['last_node_color']

        self.print_f('cal pos', verbose=2)
        all_active_nodes = self.network.new_vertex_property('bool')
        if self.first_iteration:
            all_active_nodes = active_nodes
            self.active_nodes = copy.copy(active_nodes)
        elif self.keep_inactive_nodes:
            all_active_nodes.a = self.active_nodes.a | active_nodes.a
        else:
            all_active_nodes.a = active_nodes.a

        #calc dynamic positioning
        if dynamic_pos:
            pos_tmp_net = GraphView(self.network, vfilt=all_active_nodes, efilt=active_edges)
            old_pos_update = pos_tmp_net.new_vertex_property('vector<float>')
            old_pos_abs_update = pos_tmp_net.new_vertex_property('vector<float>')
            if self.mark_new_active_nodes:
                colors = pos_tmp_net.new_vertex_property('vector<float>')
                colors.set_2d_array(np.array([np.array([0.0, 0.0, 1.0, self.max_node_alpha]) if self.active_nodes[n] else np.array([1.0, 0.0, 0.0, self.max_node_alpha]) for n in pos_tmp_net.vertices()]).T)
            for v in pos_tmp_net.vertices():
                old_pos_update[v] = self.pos[v]
                old_pos_abs_update[v] = self.pos_abs[v]
            self.pos = old_pos_update
            self.pos_abs = old_pos_abs_update

            count_new_active = 0
            for v in filter(lambda m: not self.active_nodes[m] and active_nodes[m], pos_tmp_net.vertices()):
                count_new_active += 1
                orig_abs_pos = [(self.pos[n].a, self.pos_abs[n].a) for n in filter(lambda ln: self.active_nodes[ln], v.all_neighbours())]
                if len(orig_abs_pos):
                    orig, abs_orig = zip(*orig_abs_pos)
                    rand_norm = np.random.normal(1, 0.01, 2)
                    self.pos[v] = np.array(orig).mean(axis=0) * rand_norm
                    self.pos_abs[v] = np.array(abs_orig).mean(axis=0) * rand_norm
                else:
                    self.pos[v] = []
                    self.pos_abs[v] = []
                self.print_f('mean pos:', self.pos[v], verbose=2)
            self.print_f('new active nodes:', count_new_active, '(', count_new_active / pos_tmp_net.num_vertices() * 100, '%)', verbose=2)
            try:
                new_pos = self.calc_grouped_sfdp_layout(network=pos_tmp_net, groups_vp='groups', pos=self.pos)
                self.print_f('dyn pos: updated grouped sfdp', verbose=2)
            except KeyError:
                self.print_f('dyn pos: update sfdp', verbose=2)
                new_pos = sfdp_layout(pos_tmp_net, pos=self.pos, mu=self.mu, multilevel=False, max_iter=(int(count_new_active * np.log2(count_new_active + 1))) if count_new_active > 0 else 0, epsilon=0.1)

            # calc absolute positions
            new_pos_abs = self.calc_absolute_positions(new_pos, network=pos_tmp_net)
            for v in filter(lambda lv: not self.active_nodes[lv], pos_tmp_net.vertices()):
                if not self.pos_abs[v]:
                    self.pos_abs[v] = new_pos_abs[v].a * np.random.normal(1, 0.01, 2)
                if not self.pos[v]:
                    self.pos[v] = new_pos[v].a * np.random.normal(1, 0.01, 2)
            edges_graph = pos_tmp_net
            nodes_graph_vfilt = label_largest_component(pos_tmp_net) if self.lc_only else None
            nodes_graph = GraphView(pos_tmp_net, vfilt=nodes_graph_vfilt, efilt=self.network.ep['no_edges_filt'])
        else:
            #static graph -> no repositioning
            new_pos_abs = self.pos_abs
            new_pos = self.pos

        # calc output and node size
        num_nodes = nodes_graph.num_vertices() if dynamic_pos else self.network.num_vertices()
        tmp_output_size = min(self.output_size) * 0.9
        if num_nodes < 10:
            num_nodes = 10
        max_vertex_size = np.sqrt((np.pi * (tmp_output_size / 2) ** 2) / num_nodes)
        if max_vertex_size < min_vertex_size_shrinking_factor:
            max_vertex_size = min_vertex_size_shrinking_factor
        min_vertex_size = max_vertex_size / min_vertex_size_shrinking_factor
        if len(set(v_state.a)) == 1:
            max_vertex_size -= ((max_vertex_size - min_vertex_size) / 2)
            if max_vertex_size < 1:
                max_vertex_size = 1
            min_vertex_size = max_vertex_size
        output_size = self.output_size
        vertex_size = prop_to_size(v_state, mi=min_vertex_size, ma=max_vertex_size, power=1)
        if self.first_iteration:
            old_vertex_size = prop_to_size(v_state, mi=min_vertex_size, ma=max_vertex_size, power=1)
            self.network.vp['last_node_size'] = old_vertex_size
        else:
            old_vertex_size = self.network.vp['last_node_size']

        for smoothing_step in range(smoothing):
            fac = (smoothing_step + 1) / smoothing
            old_fac = 1 - fac
            new_fac = fac
            if self.draw_fractions:
                for v in nodes_graph.vertices():
                    tmp = []
                    for mod, val in zip(list(fraction_mods[v]), list(interpolated_fraction_vals[v])):
                        if mod == 0:
                            val += stay_fraction_change[v]
                        elif mod == 1:
                            val += emerge_fraction_increase[v]
                        elif mod == -1:
                            val += vanish_fraction_reduce[v]
                        else:
                            self.print_f('ERROR: Fraction modification unknown')
                            raise Exception
                        tmp.append(val)
                    interpolated_fraction_vals[v] = tmp
                vertex_shape = "pie"
                interpolated_color = colors
            else:
                interpolated_fraction_vals = []
                vertex_shape = "circle"
                interpolated_color = nodes_graph.new_vertex_property('vector<float>')
                interpolated_color.set_2d_array(new_fac * colors.get_2d_array((0, 1, 2, 3)) + old_fac * last_node_color.get_2d_array((0, 1, 2, 3)))

            interpolated_size.a = old_fac * old_vertex_size.a + new_fac * vertex_size.a
            if dynamic_pos:
                interpolated_pos = nodes_graph.new_vertex_property('vector<float>')
                interpolated_pos.set_2d_array(old_fac * self.pos_abs.get_2d_array((0, 1)) + new_fac * new_pos_abs.get_2d_array((0, 1)))
            else:
                interpolated_pos = self.pos_abs

            if edges_graph is not None and ((smoothing_step == 0 or self.edge_blending) or dynamic_pos):
                eorder = edges_graph.new_edge_property('float')
                eorder.a = edge_color.get_2d_array([3])[0]
                if dynamic_pos or self.edge_blending:
                    interpolated_edge_color = edges_graph.new_edge_property('vector<float>')
                    interpolated_edge_color.set_2d_array(last_edge_color.get_2d_array((0, 1, 2, 3)) * old_fac + edge_color.get_2d_array((0, 1, 2, 3)) * new_fac)
                else:
                    interpolated_edge_color = edge_color
                self.print_f('draw edgegraph', verbose=2)
                if nodes_graph.num_vertices() > 0:
                    graph_draw(edges_graph, output=self.edges_filename, output_size=output_size, pos=interpolated_pos, fit_view=False, vorder=interpolated_size, vertex_size=0, vertex_fill_color=self.bg_color, vertex_color=self.bg_color, edge_pen_width=1,
                               edge_color=interpolated_edge_color, eorder=eorder, vertex_pen_width=0.0, bg_color=self.bg_color)
                else:
                    empty_img = Image.new("RGBA", self.output_size, tuple([int(i*255) for i in self.bg_color]))
                    empty_img.save(self.edges_filename, 'PNG')
                self.print_f('ok', verbose=2)
                plt.close('all')

            filename = self.generate_filename(self.output_filenum)
            self.print_f('draw nodegraph', verbose=2)
            if nodes_graph.num_vertices() > 0:
                graph_draw(nodes_graph, fit_view=False, pos=interpolated_pos, vorder=interpolated_size, vertex_size=interpolated_size, vertex_pie_fractions=interpolated_fraction_vals,
                           vertex_pie_colors=interpolated_color, vertex_fill_color='blue' if self.draw_fractions else interpolated_color, vertex_shape=vertex_shape, output=filename,
                           output_size=output_size, vertex_pen_width=0.0, vertex_color='blue' if self.draw_fractions else interpolated_color)
            else:
                empty_img = Image.new("RGBA", self.output_size, tuple([int(i*255) for i in self.bg_color]))
                empty_img.save(filename, 'PNG')
            self.output_filenum += 1
            self.print_f('ok', verbose=2)
            generated_files.append(filename)
            plt.close('all')

            bg_img = Image.open(self.edges_filename)
            fg_img = Image.open(filename)
            bg_img.paste(fg_img, None, fg_img)
            bg_img.save(filename, 'PNG')

        self.network.vp['last_node_size'] = vertex_size
        self.network.ep['edge_color'] = edge_color
        self.network.vp['last_node_color'] = copy.copy(colors)
        if self.draw_fractions:
            self.network.vp['last_fraction_map'] = copy.copy(fraction_map)
        if dynamic_pos:
            new_pos_ar = new_pos.get_2d_array((0, 1)).T
            new_pos_ar = new_pos_ar[list(map(int, nodes_graph.vertices()))]
            mean_pos = new_pos_ar.mean(axis=0)
            new_pos.set_2d_array((new_pos_ar - mean_pos).T)
            self.pos = new_pos
            self.pos_abs = new_pos_abs
            self.active_nodes.a = all_active_nodes.a
        self.first_iteration = False
        return generated_files

    def label_output(self):
        num_generated_images = sum(map(len, self.generate_files.values()))
        text_font = ImageFont.truetype("/usr/share/fonts/truetype/liberation/LiberationSans-Regular.ttf", int(round(self.output_size[1] * 0.05)))
        blending = len(self.generate_files[sorted(self.generate_files.keys())[2]]) > 10
        self.print_f('label', num_generated_images, 'pictures', ('with blending' if blending else ''))
        label_pos = (self.output_size[0] - (0.25 * self.output_size[0]), self.output_size[1] - (0.1 * self.output_size[1]))
        label_img_size = self.output_size
        label_im_bgc = (255, 255, 255, 0)
        for idx, (label, files) in enumerate(self.generate_files.iteritems()):
            # if int((idx / len(self.generate_files.keys())) * 10) > int(((idx - 1) / len(self.generate_files.keys())) * 10):
            # print 10 - int((idx / len(self.generate_files.keys())) * 10),
            label = str(label)
            if blending:
                # blend-in and out label
                alpha_values = np.array([(len(files) / 2) - abs(smoothing_step - (len(files) / 2)) for smoothing_step in range(len(files))])
                alpha_values /= alpha_values.max() * (1 + (1 / len(files)))
                alpha_values = (np.array([min(1, i) for i in alpha_values]) * 255).astype('int')
                label_img = None
                for img_idx, img_fname in enumerate(files):
                    if img_idx == 0 or (img_idx > 0 and alpha_values[img_idx-1] != alpha_values[img_idx]):
                        label_img = Image.new("RGBA", label_img_size, label_im_bgc)
                        label_drawer = ImageDraw.Draw(label_img)
                        label_drawer.text(label_pos, label, font=text_font, fill=(0, 0, 0, alpha_values[img_idx]))
                    img = Image.open(img_fname)
                    img.paste(label_img, (0, 0), label_img)
                    img.save(img_fname)
            else:
                # no blending of label
                label_img = Image.new("RGBA", self.output_size, (255, 255, 255, 0))
                label_drawer = ImageDraw.Draw(label_img)
                label_drawer.text(label_pos, label, font=text_font, fill=(0, 0, 0, 255))
                for img_idx, img_fname in enumerate(files):
                    img = Image.open(img_fname)
                    img.paste(label_img, (0, 0), label_img)
                    img.save(img_fname)

    def create_video(self, fps, delete_pictures=True):
        if self.filename_basename.endswith('.png'):
            file_basename = self.filename_basename[:-4]
        else:
            file_basename = self.filename_basename
        if _platform == "linux" or _platform == "linux2":
            num_generated_images = sum(map(len, self.generate_files.values()))
            with open(os.devnull, "w") as devnull:
                self.print_f('create movie of', num_generated_images, 'pictures', verbose=1)
                p_call = ['ffmpeg', '-framerate', str(fps), '-i', self.filename_folder + '/' + self.tmp_folder_name + '%06d' + file_basename + '.png', '-framerate',
                          str(fps), '-r', str(fps), '-c:v', 'libx264', '-y', '-pix_fmt', 'yuv420p',
                          self.filename_folder + '/' + file_basename.strip('_') + '.avi']
                self.print_f('call:', p_call, verbose=2)
                exit_status = subprocess.check_call(p_call, stdout=devnull, stderr=devnull)
                if exit_status == 0 and delete_pictures:
                    self.print_f('delete pictures...', verbose=1)
                    _ = subprocess.check_call(['rm ' + str(self.filename_folder + '/' + self.tmp_folder_name + '*' + file_basename + '.png')], shell=True, stdout=devnull)


