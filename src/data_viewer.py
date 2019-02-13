from matplotlib import pyplot


class DataViewer:

    def __init__(self, *counts):
        self.counts = counts

    def plot_counts_vs_attributes(self, attr, title, xtick=None, save_as_pdf=True):
        pyplot.figure()
        pyplot.title(title)

        colors = "grby"
        for count_object, color in zip(self.counts, colors):
            pyplot.scatter(attr, count_object[0], c=color, s=5, label=count_object[1])

        if xtick:
            pyplot.xticks(xtick)

        pyplot.legend()

        if save_as_pdf:
            pyplot.savefig("./figs/{}.pdf".format(title))
            pyplot.close()
        else:
            pyplot.show()
