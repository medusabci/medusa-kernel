# Python modules
import argparse
import os
# Medusa modules
from medusa.meeg import EEGChannelSet

# External modules
import numpy as np
from vispy import scene, app
from vispy.io import imread, load_data_file, read_mesh
from vispy.scene.visuals import Mesh, Markers, Text, Graph
from vispy.scene import transforms
from vispy.visuals.filters import TextureFilter
from medusa.meeg import EEGChannelSet


class TridimentionalBrain():
    def __init__(self, bg_color='black', text_color='white', cameras=None,
                 translucent=None, subplots=None, models=None, names=None):

        self.subplots = subplots
        self.bg_color = bg_color
        self.text_color = text_color
        self.cameras = cameras
        self.translucent = translucent
        self.models = models
        self.names = names

        # Initialize attributes
        self.canvas = None
        self.views = []
        self.grid = None
        self.standards = []
        self.sources = []
        self.brain_visuals = []
        self.markers = None
        self.labels_text = None
        self.connections_coords = None
        self.connections_values = None
        self.lines = None
        self.n_subplots = None

        self.__check_subplot()

        # Set canvas and view
        self.__set_canvas()

        # Set alpha
        self.__set_alpha()

        # Set brain models
        self.__set_models()

        # Set brain visuals
        self.__set_brain_visual()

        # Initializing methods
        self.__initialize_connections()

        self.canvas.show()

    def __check_subplot(self):
        try:
            if self.subplots is not None:
                assert isinstance(self.subplots, tuple)
                assert len(self.subplots) == 2
                self.n_subplots = self.subplots[0] * self.subplots[1]
            else:
                self.n_subplots = 1
                self.subplots = (1, 1)
        except Exception as ex:
            print(ex)

    def __set_alpha(self):
        try:
            if self.translucent is None:
                self.translucent = np.zeros(self.n_subplots, dtype=bool)
            else:
                assert len(self.translucent) == self.n_subplots
        except Exception as ex:
            print(ex)

    def __set_models(self):
        try:
            if self.models is None:
                self.models = ['B1']
                self.models = list(np.tile(self.models, self.n_subplots))
            else:
                assert len(self.models) == self.n_subplots
        except Exception as ex:
            print(ex)

    def __set_cameras(self):
        try:
            self.cameras = []
            for i in range(self.n_subplots):
                self.cameras.append('turntable')
        except Exception as ex:
            print(ex)

    def __set_canvas(self):
        try:
            self.canvas = scene.SceneCanvas(keys='interactive', bgcolor=self.bg_color,
                                            size=(850, 850),show=True)
            if self.cameras is not None:
                assert len(self.cameras) == self.n_subplots
            else:
                self.__set_cameras()
            if self.subplots is None:
                view = self.canvas.central_widget.add_view()
                view.camera = self.cameras[0]
                self.views.append(view)
            else:
                if self.names is not None:
                    assert len(self.names) == self.n_subplots
                self.grid = self.canvas.central_widget.add_grid(margin=10)
                self.grid.spacing = 0
                _view_idx = 0
                for row in range(self.subplots[0]):
                    for col in range(self.subplots[1]):
                        if self.n_subplots is None:
                            view = self.grid.add_view(row=row, col=col)
                        else:
                            if self.names is not None:
                                assert isinstance(self.names[_view_idx], str)
                                title = scene.Label(self.names[_view_idx], color=self.text_color)
                                title.height_max = 60
                                self.grid.add_widget(title, row=row, col=col, col_span=1)
                                view = self.grid.add_view(row=row + 1, col=col)
                            else:
                                view = self.grid.add_view(row=row, col=col)
                        view.camera = self.cameras[_view_idx]
                        self.views.append(view)
                        _view_idx += 1
        except Exception as ex:
            print(ex)

    def __set_brain_visual(self):
        try:
            parser = argparse.ArgumentParser()
            parser.add_argument('--shading', default='flat',
                                choices=['none', 'flat', 'smooth'],
                                help="shading mode")
            args, _ = parser.parse_known_args()
            shading = None if args.shading == 'none' else args.shading

            for i in range(self.n_subplots):
                path = os.path.join(os.path.dirname(__file__), 'templates/{}.npz'.format(self.models[i]))
                model = np.load(path)
                vertices, faces = model.f.vertices, model.f.faces

                # TODO This step is temporary
                vertices[:, 0] = 0.85 * vertices[:, 0] / np.max(vertices[:, 2])
                vertices[:, 1] = 0.85 * vertices[:, 1] / np.max(vertices[:, 2])
                vertices[:, 2] = 0.85 * vertices[:, 2] / np.max(vertices[:, 2])

                # Adapt the depth to the scale of the mesh to avoid rendering artefacts.
                self.views[i].camera.depth_value = 10 * (vertices.max() - vertices.min())

                # Define transparency of each brain
                if self.translucent[i]:
                    alpha = 0.3
                else:
                    alpha = 1
                mesh = Mesh(vertices, faces, shading=shading, color=(1, 1, 1, alpha))
                mesh.shading_filter.shininess = 1e+2
                self.brain_visuals.append(mesh)
                self.attach_headlight(mesh, self.views[i], self.canvas)
        except Exception as ex:
            print(ex)

    def add_brains(self):
        try:
            for view_idx in range(len(self.views)):
                self.views[view_idx].add(self.brain_visuals[view_idx])
        except Exception as ex:
            print(ex)

    def attach_headlight(self, mesh, view, canvas):
        light_dir = (1, 0, 1, 1)
        mesh.shading_filter.light_dir = light_dir[:3]
        initial_light_dir = view.camera.transform.imap(light_dir)

        @view.scene.transform.changed.connect
        def on_transform_change(event):
            transform = view.camera.transform
            mesh.shading_filter.light_dir = transform.map(initial_light_dir)[:3]

    def set_markers(self, locs, sub_plot):
        try:
            if self.markers is None:
                self.__init_markers()
            markers = Markers()
            markers.set_data(locs)
            self.markers[sub_plot[0]][sub_plot[1]] = markers

            _view_idx = self.__calculate_subplot_idx(sub_plot)
            self.views[_view_idx].add(markers)
        except Exception as ex:
            print(ex)

    def set_labels(self, labels,locs, sub_plot):
        try:
            if self.labels_text is None:
                self.__init_labels()

            # Move the labels over the channels
            text_coord = locs.copy()
            text_coord[:,2] += 0.1
            labels_text = Text(labels,color = self.text_color,bold=True,
                               pos = text_coord,font_size=50)
            self.labels_text[sub_plot[0]][sub_plot[1]] = labels_text

            _view_idx = self.__calculate_subplot_idx(sub_plot)
            self.views[_view_idx].add(labels_text)
        except Exception as ex:
            print(ex)

    def __set_conn_color(self,sub_plot,threshold):
        try:
            # Mask under threshold values
            # TODO Arreglar tema de colores/valores
            mask = np.ones(len(self.connections_values[sub_plot[0]][sub_plot[1]]), dtype=bool)
            mask[np.where(self.connections_values[sub_plot[0]][sub_plot[1]] < threshold)] = 0

            self.connections_values[sub_plot[0]][sub_plot[1]] = self.connections_values[sub_plot[0]][sub_plot[1]][mask]
            self.connections_coords_mutable[sub_plot[0]][sub_plot[1]] = \
            self.connections_coords_invariant[sub_plot[0]][sub_plot[1]][mask]

            self.connections_values[sub_plot[0]][sub_plot[1]] = np.tile(self.connections_values[sub_plot[0]]
                                                                        [sub_plot[1]], (2, 1)).reshape(
                2 * len(self.connections_values[
                            sub_plot[0]][sub_plot[1]]), order='F')
            color = np.zeros((len(self.connections_values[sub_plot[0]][sub_plot[1]]), 4), dtype=np.float32)
            color[:, 0] = self.connections_values[sub_plot[0]][sub_plot[1]]
            color[:, -1] = 1
            return color
        except Exception as ex:
            print(ex)

    def set_connections(self, adj_mat, locs, sub_plot=None, threshold=0.5,
                        plot_markers=True, labels=None, plot_labels=False):
        try:
            if sub_plot is None and self.n_subplots == 1:
                sub_plot = (0, 0)
            assert isinstance(sub_plot, tuple)
            assert sub_plot[0] <= self.subplots[0] and sub_plot[1] <= self.subplots[1]

            if plot_markers:
                self.set_markers(locs, sub_plot)
            if plot_labels:
                if labels is None:
                    print("Labels could not been added because labels parameter"
                          "is None")
                else:
                    assert all(isinstance(elem, str) for elem in labels)
                    self.set_labels(labels, locs, sub_plot)

            # Extract connectivity values
            self.connections_values[sub_plot[0]][sub_plot[1]] = self.__extract_conn_values(adj_mat)

            # Set connections matrix
            self.connections_coords_invariant[sub_plot[0]][sub_plot[1]] = self.__set_connections_coords(locs,
                                                                                                        self.connections_values
                                                                                                        [sub_plot[0]][
                                                                                                            sub_plot[
                                                                                                                1]])

            color = self.__set_conn_color(sub_plot,threshold)

            # Draw line conecctions
            if self.subplots is None:
                _view_idx = 0
            else:
                _view_idx = self.__calculate_subplot_idx(sub_plot)
            self.lines[sub_plot[0]][sub_plot[1]] = scene.Line(antialias=True, parent = self.views[_view_idx].scene)
            self.lines[sub_plot[0]][sub_plot[1]].set_data( pos=self.connections_coords_mutable[sub_plot[0]][sub_plot[1]],
                color=color,  width=3,)
        except Exception as ex:
            print(ex)

    def update_connections(self,adj_mat,sub_plot,threshold):
        try:
            if sub_plot is None:
                sub_plot = (0,0)
            if threshold is None:
                threshold = 0.5
            # Extract connectivity values
            self.connections_values[sub_plot[0]][sub_plot[1]] = self.__extract_conn_values(adj_mat)
            color = self.__set_conn_color(sub_plot,threshold)
            _view_idx = self.__calculate_subplot_idx(sub_plot)

            self.lines[sub_plot[0]][sub_plot[1]].parent = None

            self.lines[sub_plot[0]][sub_plot[1]] = scene.Line(antialias=True, parent=self.views[_view_idx].scene)
            self.lines[sub_plot[0]][sub_plot[1]].set_data(pos=self.connections_coords_mutable[sub_plot[0]][sub_plot[1]],
                                                          color=color, width=3, )
            self.canvas.update()
        except Exception as ex:
            print(ex)


    def __calculate_subplot_idx(self,sub_plot):
        try:
            _view_idx = sub_plot[1] + self.subplots[1] * sub_plot[0]
            return _view_idx
        except Exception as ex:
            print(ex)

    def __initialize_connections(self):
        try:
            subplots = self.subplots
            if self.n_subplots is None:
                subplots = (1, 1)
            self.connections_coords_invariant = np.empty(shape=subplots + (0,)).tolist()
            self.connections_coords_mutable = np.empty(shape=subplots + (0,)).tolist()
            self.connections_values = np.empty(shape=subplots + (0,)).tolist()
            self.lines = np.empty(shape=subplots + (0,)).tolist()
        except Exception as ex:
            print(ex)

    def __init_markers(self):
        try:
            subplots = self.subplots
            self.markers = np.empty(shape=subplots + (0,)).tolist()
        except Exception as ex:
            print(ex)

    def __init_labels(self):
        try:
            subplots = self.subplots
            self.labels_text = np.empty(shape=subplots + (0,)).tolist()
        except Exception as ex:
            print(ex)

    def __extract_conn_values(self, adj_mat):
        try:
            values = np.triu(adj_mat, 1)
            values = values[np.where(values != 0)]
        except Exception as ex:
            print(ex)
        return values

    def __set_connections_coords(self, locs, connections_values):
        try:
            connections_coords = np.empty((len(connections_values), 2, 3))
            value_idx = 0
            for i in range(len(locs)):
                for j in range(i + 1, len(locs)):
                    connections_coords[value_idx, 0, 0] = locs[i, 0]
                    connections_coords[value_idx, 0, 1] = locs[i, 1]
                    connections_coords[value_idx, 0, 2] = locs[i, 2]
                    connections_coords[value_idx, 1, 0] = locs[j, 0]
                    connections_coords[value_idx, 1, 1] = locs[j, 1]
                    connections_coords[value_idx, 1, 2] = locs[j, 2]
                    value_idx += 1
        except Exception as ex:
            print(ex)

        return connections_coords


if __name__ == '__main__':
    from vispy import app

    # Set canvas
    triplot = TridimentionalBrain(subplots=(2,1))

    # Define channel set and its coord
    channel_set = EEGChannelSet(dim='3D', coord_system='cartesian')
    channel_set.set_standard_montage(standard='10-20')
    channel_coord = np.zeros((len(channel_set.channels), 3))
    for ch_idx, channel in enumerate(channel_set.channels):
        channel_coord[ch_idx, 0] = channel['x']
        channel_coord[ch_idx, 1] = channel['y'] - 0.2
        channel_coord[ch_idx, 2] = channel['z']
    adj_mat = np.random.random((len(channel_set.channels), len(channel_set.channels)))
    triplot.set_connections(adj_mat, channel_coord, threshold=[0.9], sub_plot=(0, 0), plot_labels=True,labels=channel_set.l_cha)
    triplot.set_connections(adj_mat, channel_coord, threshold=[0.4], sub_plot=(0, 1), plot_labels=True,labels=channel_set.l_cha)
    triplot.add_brains()
    triplot.canvas.show()
    app.run()
