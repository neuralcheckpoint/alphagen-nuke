"""Register Alpha Gen in the Medratools toolbar."""

import nuke

import alphagen_nuke

nuke.addOnScriptLoad(alphagen_nuke.upgrade_quality_nodes)
alphagen_nuke.upgrade_quality_nodes()

_toolbar = nuke.menu("Nodes")
_tools = _toolbar.findItem("Medratools") or _toolbar.addMenu("Medratools", icon="medratools.png")
_tools.addCommand("Alpha Gen", "import alphagen_nuke; alphagen_nuke.create()", icon="medratools.png")
