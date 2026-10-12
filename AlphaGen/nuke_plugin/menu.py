"""Register Alpha Gen in the Medratools toolbar."""

import nuke

import alphagen_nuke

_toolbar = nuke.menu("Nodes")
_tools = _toolbar.findItem("Medratools") or _toolbar.addMenu("Medratools", icon="medratools.png")
_tools.addCommand("Alpha Gen", "import alphagen_nuke; alphagen_nuke.create()", icon="medratools.png")
