"""Alpha Gen node for Nuke."""

import json
import os
import re
import subprocess
import sys

import nuke

PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
_SHARED = os.path.join(os.path.dirname(PROJECT_ROOT), "aitools", "nuke")
if _SHARED not in sys.path:
    sys.path.insert(0, _SHARED)
import aitools_nuke  # noqa: E402
from aitools_engine.i18n import tr  # noqa: E402  (on sys.path through aitools_nuke)

TITLE = "Alpha Gen"
NODE_VERSION = 1
RESOLUTIONS = [
    ("544 (recommended)", 544),
    ("768", 768),
    ("1088 (high VRAM)", 1088),
]
OUTPUTS = ["alpha", "Alpha Gen layer"]
SOURCES = ["Auto", "Render input"]
SUBJECT_POINT = re.compile(r"^subject_point_(\d+)$")
MAX_POINTS = 8


def output_layer(node):
    return re.sub(r"[^a-z0-9_.]", "_", node.name().lower()) + ".alpha"


def register_layer(node):
    layer = output_layer(node)
    nuke.tcl("add_layer {{{} {}}}".format(layer.rsplit(".", 1)[0], layer))


def create():
    selected = nuke.selectedNodes()
    group = nuke.createNode("Group", inpanel=False)
    index = 1
    while nuke.toNode("AlphaGen{}".format(index)) is not None:
        index += 1
    group.setName("AlphaGen{}".format(index))
    group["tile_color"].setValue(0x685EAEFF)
    register_layer(group)
    _add_knobs(group)
    _build_internals(group)
    if len(selected) == 1:
        group.setInput(0, selected[0])
    source = group.input(0)
    if source is not None:
        group["range_first"].setValue(int(source.firstFrame()))
        group["range_last"].setValue(int(source.lastFrame()))
    else:
        group["range_first"].setValue(int(nuke.root().firstFrame()))
        group["range_last"].setValue(int(nuke.root().lastFrame()))
    group.showControlPanel()
    return group


def _build_internals(group):
    layer = output_layer(group)
    with group:
        source = nuke.nodes.Input(name="Source")
        cache = aitools_nuke.add_cache_read()
        # Before Compute the cache is black without the layer; keep the output valid so the preview shows.
        cache = nuke.nodes.AddChannels(name="CacheLayer", inputs=[cache], channels=layer)
        into_alpha = nuke.nodes.Copy(name="CopyAlpha", inputs=[source, cache])
        into_alpha["from0"].setValue(layer)
        into_alpha["to0"].setValue("rgba.alpha")
        into_alpha["disable"].setExpression("parent.output != 0")
        into_layer = nuke.nodes.Copy(name="CopyLayer", inputs=[into_alpha, cache])
        into_layer["from0"].setValue(layer)
        into_layer["to0"].setValue(layer)
        into_layer["disable"].setExpression("parent.output != 1")
        nuke.nodes.Output(inputs=[_add_preview_tint(into_layer)])


def _frame_read(name, pattern_knob):
    """A Read of one of the node's EXR sequences, black outside the range or before it exists."""
    read = nuke.nodes.Read(name=name)
    read["file"].setValue("[knob parent.{}]".format(pattern_knob))
    read["file_type"].setValue("exr")
    for knob in ("first", "origfirst"):
        read[knob].setExpression("parent.range_first")
    for knob in ("last", "origlast"):
        read[knob].setExpression("parent.range_last")
    for knob in ("before", "after", "on_error"):
        read[knob].setValue("black")
    read["raw"].setValue(True)
    return read


def _add_preview_tint(upstream):
    """Selection overlay, off unless show_preview is on.

    PeopleMap (subjects.id, from Analyze people) is picked by an expression the
    points rewrite, so the overlay follows a dragged point without running a
    model; SubjectPreview (subject.alpha) adds what SAM tracked for points off
    the map. Picked people get a fill and an outline in their own colour, the
    other people of the map a faint fill, SAM selections red.
    """
    nuke.tcl("add_layer {subject subject.alpha}")
    nuke.tcl("add_layer {subjects subjects.id}")
    nuke.tcl("add_layer {agsel agsel.pick agsel.id agsel.edge}")
    people = nuke.nodes.AddChannels(name="PeopleLayer", inputs=[_frame_read("PeopleMap", "analysis_pattern")],
                                    channels="subjects")
    sam = nuke.nodes.AddChannels(name="PreviewLayer", inputs=[_frame_read("SubjectPreview", "preview_pattern")],
                                 channels="subject")
    merged = nuke.nodes.Copy(name="PreviewMerge", inputs=[people, sam])
    merged["from0"].setValue("subject.alpha")
    merged["to0"].setValue("subject.alpha")
    merged["disable"].setExpression("!parent.preview_fresh")  # SAM preview of other points is stale
    pick = nuke.nodes.Expression(name="PeoplePick", inputs=[merged])
    for index, (channel, expression) in enumerate((
        ("agsel.pick", "subject.alpha"), ("agsel.id", "subjects.id"), ("agsel.edge", "subject.alpha"),
    )):
        pick["channel{}".format(index)].setValue(channel)
        pick["expr{}".format(index)].setValue(expression)
    erode = nuke.nodes.FilterErode(name="PickErode", inputs=[pick])
    erode["channels"].setValue("agsel.edge")
    erode["size"].setValue(3)
    edge = nuke.nodes.Expression(name="PickEdge", inputs=[erode])
    edge["channel0"].setValue("agsel.edge")
    edge["expr0"].setValue("agsel.pick - agsel.edge")
    copy = nuke.nodes.Copy(name="PreviewCopy", inputs=[upstream, edge])
    copy["channels"].setValue("agsel")
    tint = nuke.nodes.Expression(name="PreviewTint", inputs=[copy])
    tint["temp_name0"].setValue("k")
    tint["temp_expr0"].setValue("fmod(agsel.id + 1, 6)")  # person 1 blue, 2 yellow...; red stays for SAM
    tint["temp_name1"].setValue("w")
    tint["temp_expr1"].setValue(
        "agsel.edge > 0.5 ? 0.9 : agsel.pick > 0.5 ? 0.45 : agsel.id > 0.5 ? 0.15 : 0.6 * agsel.pick")
    for index, colours in enumerate(("(k == 0 || k == 3 || k == 4)", "(k == 1 || k == 3 || k == 5)",
                                     "(k == 2 || k == 4 || k == 5)")):
        sam_colour = "1" if index == 0 else "0.15"
        channel = "rgb"[index]
        tint["expr{}".format(index)].setValue(
            "{c} * (1 - w) + w * (agsel.id < 0.5 ? {sam} : {colours} ? 1 : 0.2)".format(
                c=channel, sam=sam_colour, colours=colours))
    clean = nuke.nodes.Remove(name="PreviewClean", inputs=[tint], channels="agsel")
    for node in (copy, tint, clean):
        node["disable"].setExpression("!parent.show_preview")
    return clean


def _subject_numbers(node):
    return sorted(
        int(match.group(1))
        for name in node.knobs()
        for match in [SUBJECT_POINT.match(name)]
        if match
    )


ROW_KNOBS = ("subject_point_{}", "subject_id_{}", "subject_exclude_{}", "subject_remove_{}")


def _add_subject_knobs(node, number):
    """One hidden point row; "+ point" reveals it (Nuke can only append knobs at the end of a panel)."""
    enabled = nuke.Boolean_Knob("subject_enabled_{}".format(number), "")
    enabled.setVisible(False)
    node.addKnob(enabled)

    frame_knob = nuke.Int_Knob("subject_frame_{}".format(number), "")
    frame_knob.setValue(-1)
    frame_knob.setVisible(False)
    node.addKnob(frame_knob)

    xy = nuke.XY_Knob("subject_point_{}".format(number), tr("point {}", number))
    xy.setTooltip(tr("Drag the point in the Viewer; its frame is saved when you move it."))
    node.addKnob(xy)

    owner = nuke.Int_Knob("subject_id_{}".format(number), tr("subject"))
    owner.setValue(1)
    owner.clearFlag(nuke.STARTLINE)
    owner.setTooltip(tr("Points with the same subject number are clicks on the same person or object."))
    node.addKnob(owner)

    exclude = nuke.Boolean_Knob("subject_exclude_{}".format(number), tr("exclude"))
    exclude.clearFlag(nuke.STARTLINE)
    exclude.setTooltip(tr("Leave this area out of the subject, e.g. another person or a prop it grabbed."))
    node.addKnob(exclude)

    hit = nuke.Int_Knob("subject_hit_{}".format(number), "")
    hit.clearFlag(nuke.STARTLINE)
    hit.setVisible(False)  # person of the map under the point; 0 = none, SAM tracks the click
    node.addKnob(hit)

    remove = nuke.PyScript_Knob(
        "subject_remove_{}".format(number), tr("remove"),
        "import alphagen_nuke; alphagen_nuke.remove_subject(nuke.thisNode(), {})".format(number),
    )
    remove.clearFlag(nuke.STARTLINE)
    node.addKnob(remove)
    _show_row(node, number, False)


def _show_row(node, number, visible):
    for name in ROW_KNOBS:
        knob = node.knob(name.format(number))
        if knob is not None:
            knob.setVisible(visible)


def _active_numbers(node):
    return [n for n in _subject_numbers(node) if node["subject_enabled_{}".format(n)].value()]


def _subject_id(node, number):
    knob = node.knob("subject_id_{}".format(number))
    return int(knob.value()) if knob is not None else number


def add_subject(node, new_subject=False):
    """Show a point at the plate centre on the current frame: a click on the last subject, or a new one."""
    active = _active_numbers(node)
    free = [n for n in _subject_numbers(node) if n not in active]
    if not free:
        nuke.message(tr("{}: all {} points are in use; remove one first.", TITLE, len(active)))
        return
    ids = [_subject_id(node, number) for number in active]
    subject_id = max(ids, default=0) + 1 if new_subject else (ids[-1] if ids else 1)
    number = free[0]
    source = node.input(0)
    image_format = source.format() if source is not None else nuke.root().format()
    node["subject_id_{}".format(number)].setValue(subject_id)
    node["subject_exclude_{}".format(number)].setValue(False)
    node["subject_frame_{}".format(number)].setValue(int(nuke.frame()))
    node["subject_enabled_{}".format(number)].setValue(True)
    node["subject_point_{}".format(number)].setValue([image_format.width() / 2.0, image_format.height() / 2.0])
    _show_row(node, number, True)
    _update_hit(node, number)
    refresh_selection(node)


def remove_subject(node, number):
    node["subject_enabled_{}".format(number)].setValue(False)
    node["subject_frame_{}".format(number)].setValue(-1)
    node["subject_point_{}".format(number)].setValue([0.0, 0.0])
    _show_row(node, number, False)
    refresh_selection(node)


def reset_subjects(node):
    for number in _active_numbers(node):
        remove_subject(node, number)


def _hide_preview(node):
    # A changed prompt makes the red preview stale (nodes from before the person map).
    if "show_preview" in node.knobs():
        node["show_preview"].setValue(False)


def _has_people_map(node):
    return "analysis_pattern" in node.knobs()


def _update_hit(node, number):
    """Store which person of the map is under point ``number`` on its frame."""
    if not _has_people_map(node) or not node["analysis_pattern"].value():
        return
    x, y = node["subject_point_{}".format(number)].getValue()
    frame = int(node["subject_frame_{}".format(number)].value())
    with node:
        people = nuke.toNode("PeopleLayer")
        try:
            value = people.sample("subjects.id", int(x) + 0.5, int(y) + 0.5, frame=frame) if people is not None else 0.0
        except (RuntimeError, ValueError):
            value = 0.0
    node["subject_hit_{}".format(number)].setValue(int(round(value)))


def _hit(node, number):
    knob = node.knob("subject_hit_{}".format(number))
    return int(knob.value()) if knob is not None else 0


def _everyone(node):
    """Every person of the analysis, numbered 1..n like the map."""
    try:
        manifest = os.path.join(os.path.dirname(node["analysis_pattern"].value()), "analysis.json")
        with open(manifest) as handle:
            return list(range(1, int(json.load(handle)["people"]) + 1))
    except (OSError, ValueError, KeyError, TypeError):
        return []


def _picked_people(node):
    """People of the map kept by include points, minus those under exclude points.

    Only exclude points on people means everyone but them.
    """
    kept, dropped, sam_include = set(), set(), False
    for number in _active_numbers(node):
        person = _hit(node, number)
        excluded = node["subject_exclude_{}".format(number)].value()
        if person:
            (dropped if excluded else kept).add(person)
        elif not excluded:
            sam_include = True
    if dropped and not kept and not sam_include:
        kept = set(_everyone(node))
    return sorted(kept - dropped)


def refresh_selection(node):
    """Rewrite the overlay for the current points: instant for map people, SAM preview only if still valid."""
    if not _has_people_map(node):
        _hide_preview(node)
        return
    people = _picked_people(node)
    picked = " || ".join("subjects.id == {}".format(person) for person in people) or "0"
    with node:
        pick = nuke.toNode("PeoplePick")
        for index in (0, 2):
            pick["expr{}".format(index)].setValue("max(subject.alpha, ({}) ? 1 : 0)".format(picked))
    node["preview_fresh"].setValue(node["preview_subjects_json"].value() == json.dumps(_sam_points(node)))


def subject_knob_changed(node, knob):
    """Store the Viewer frame when a subject point is moved."""
    try:
        active = is_alphagen(node)
    except ValueError:
        return  # Nuke also invokes knobChanged while detaching a deleted node.
    if not active:
        return
    if knob.name().startswith(("subject_id_", "subject_exclude_")):
        refresh_selection(node)
        return
    match = SUBJECT_POINT.match(knob.name())
    if not match:
        return
    number = int(match.group(1))
    frame_name = "subject_frame_{}".format(number)
    enabled_name = "subject_enabled_{}".format(number)
    if frame_name in node.knobs() and enabled_name in node.knobs():
        point = node[knob.name()].getValue()
        if not node[enabled_name].value() and tuple(point) == (0.0, 0.0):
            return
        node[frame_name].setValue(int(nuke.frame()))
        node[enabled_name].setValue(True)
        _update_hit(node, number)
        refresh_selection(node)


def _section(group, name, label, closed=False):
    """Heading line, or a collapsible group (closed by default) for settings most shots never touch."""
    if closed:
        group.addKnob(nuke.Tab_Knob(name, label, nuke.TABBEGINCLOSEDGROUP))
    else:
        group.addKnob(nuke.Text_Knob(name, "<b>{}</b>".format(label)))


def _end_section(group, name):
    group.addKnob(nuke.Tab_Knob(name, "", nuke.TABENDGROUP))


def _add_knobs(group):
    group.addKnob(nuke.Tab_Knob("alphagen", "Alpha Gen"))
    # Run first: status and the two actions stay on screen however many points a shot has.
    group.addKnob(nuke.Text_Knob("status", tr("status"), tr("not computed")))
    compute = nuke.PyScript_Knob(
        "compute", tr("Compute"), "import alphagen_nuke; alphagen_nuke.compute(nuke.thisNode())",
    )
    compute.setFlag(nuke.STARTLINE)
    group.addKnob(compute)
    recompute = nuke.PyScript_Knob(
        "recompute", tr("Recompute all"), "import alphagen_nuke; alphagen_nuke.compute(nuke.thisNode(), force=True)",
    )
    recompute.clearFlag(nuke.STARTLINE)
    group.addKnob(recompute)
    add_backend_knobs(group)

    _section(group, "divider_input", tr("Input"))
    source = nuke.Enumeration_Knob("source_mode", tr("source"), SOURCES)
    source.setTooltip(tr(
        "Auto reads a connected Read sequence directly. Otherwise, or with Render input, "
        "the Source input is rendered to EXR before Alpha Gen runs."
    ))
    group.addKnob(source)
    first = nuke.Int_Knob("range_first", tr("frame range"))
    last = nuke.Int_Knob("range_last", "")
    last.clearFlag(nuke.STARTLINE)
    group.addKnob(first)
    group.addKnob(last)
    from_input = nuke.PyScript_Knob(
        "range_from_input", tr("from input"),
        "import alphagen_nuke; alphagen_nuke.range_from_input(nuke.thisNode())",
    )
    from_input.clearFlag(nuke.STARTLINE)
    group.addKnob(from_input)
    output = nuke.Enumeration_Knob("output", tr("output"), OUTPUTS)
    output.setTooltip(tr("Put the generated matte in rgba.alpha or in a node-specific layer."))
    group.addKnob(output)

    _section(group, "divider_subjects", tr("Subjects"))
    group.addKnob(nuke.Text_Knob("subjects_label", "", tr(
        "Without points Alpha Gen picks the subject itself.<br>"
        "To choose: Analyze people once (~15 s), then drag a point onto each person to keep. "
        "Tick exclude to drop one.<br>"
        "A point off the people (a prop, someone missed) is a SAM click: check it with Preview selection (red)."
    )))
    analyze = nuke.PyScript_Knob(
        "analyze_people", tr("Analyze people"),
        "import alphagen_nuke; alphagen_nuke.analyze(nuke.thisNode())",
    )
    analyze.setFlag(nuke.STARTLINE)
    analyze.setTooltip(tr("Find the people on the current frame and track them over the range (runs once, then exits)."))
    group.addKnob(analyze)
    for name in ("analysis_pattern", "preview_subjects_json"):
        hidden = nuke.String_Knob(name, "")
        hidden.clearFlag(nuke.STARTLINE)  # hidden knobs must not break the button row
        hidden.setVisible(False)
        group.addKnob(hidden)
    fresh = nuke.Boolean_Knob("preview_fresh", "")
    fresh.clearFlag(nuke.STARTLINE)
    fresh.setVisible(False)
    group.addKnob(fresh)
    preview_button = nuke.PyScript_Knob(
        "preview_subjects", tr("Preview selection"),
        "import alphagen_nuke; alphagen_nuke.preview(nuke.thisNode())",
    )
    preview_button.clearFlag(nuke.STARTLINE)
    group.addKnob(preview_button)
    show = nuke.Boolean_Knob("show_preview", tr("show selection"))
    show.clearFlag(nuke.STARTLINE)
    show.setTooltip(tr("Shows the selection over this node's output. Turn it off before rendering."))
    group.addKnob(show)
    preview_cache = nuke.File_Knob("preview_pattern", "")
    preview_cache.clearFlag(nuke.STARTLINE)
    preview_cache.setVisible(False)
    group.addKnob(preview_cache)
    add = nuke.PyScript_Knob(
        "add_subject", tr("+ point"),
        "import alphagen_nuke; alphagen_nuke.add_subject(nuke.thisNode())",
    )
    add.setFlag(nuke.STARTLINE)
    add.setTooltip(tr("Add another click on the last subject, at the plate centre on this frame."))
    group.addKnob(add)
    add_new = nuke.PyScript_Knob(
        "add_new_subject", tr("+ subject"),
        "import alphagen_nuke; alphagen_nuke.add_subject(nuke.thisNode(), new_subject=True)",
    )
    add_new.clearFlag(nuke.STARTLINE)
    add_new.setTooltip(tr("Add a point for another person or object."))
    group.addKnob(add_new)
    reset = nuke.PyScript_Knob(
        "reset_subjects", tr("reset subjects"),
        "import alphagen_nuke; alphagen_nuke.reset_subjects(nuke.thisNode())",
    )
    reset.clearFlag(nuke.STARTLINE)
    reset.setTooltip(tr("Remove all points; Alpha Gen returns to automatic subject selection."))
    group.addKnob(reset)
    # shortcut: fixed pool of points; raise MAX_POINTS if a shot needs more clicks.
    for number in range(1, MAX_POINTS + 1):
        _add_subject_knobs(group, number)


    _section(group, "advanced", tr("Advanced"), closed=True)
    resolution = nuke.Enumeration_Knob(
        "resolution", tr("processing resolution"), [label for label, _ in RESOLUTIONS]
    )
    resolution.setTooltip(tr(
        "Short edge used by Alpha Gen. 544 is the safe starting point on a 24 GB GPU; "
        "a Subject + overlap simulation had a 7.7 GiB backend budget (7.3 GiB free plus 0.44 GiB resident). "
        "A physical 8 GB GPU was not tested; 32 GB of host RAM is a cautious starting point. "
        "Higher settings use substantially more VRAM."
    ))
    group.addKnob(resolution)
    low_vram = nuke.Boolean_Knob("low_vram", tr("small VAE tiles"))
    low_vram.setTooltip(tr(
        "Automatic on physical GPUs up to 8 GB. Enable this on larger GPUs constrained by other apps. "
        "Small tiles can change matte edges; the unchecked 24 GB path keeps the original decode."
    ))
    group.addKnob(low_vram)
    group.addKnob(nuke.Int_Knob("seed", "seed"))
    group["seed"].setValue(1234)
    group["seed"].setTooltip(tr("Fixed seed keeps recomputes reproducible."))

    cache = nuke.File_Knob("cache_pattern", "cache")
    cache.setTooltip(tr("EXR sequence written by Compute; empty uses a folder beside the script."))
    group.addKnob(cache)

    _end_section(group, "advanced_end")

    _section(group, "about", tr("About"), closed=True)
    group.addKnob(nuke.Text_Knob("info", "", tr(
        "LTX-2.5 Alpha Gen matte; apply it as the alpha of the original plate.<br>"
        "Subject points need the optional SAM 2.1 setup (~1.2 GB, separate CUDA environment). "
        "The SAM mask only crops the matte; it cannot recover a missed subject.<br>"
        "Long shots run in overlapping chunks of up to 121 frames with blended boundaries."
    )))
    _end_section(group, "about_end")
    version = nuke.Int_Knob("alphagen_version", "")
    version.setValue(NODE_VERSION)
    version.setVisible(False)
    group.addKnob(version)

    group["knobChanged"].setValue(
        "import alphagen_nuke; alphagen_nuke.subject_knob_changed(nuke.thisNode(), nuke.thisKnob())"
    )


def is_alphagen(node):
    return node is not None and "alphagen_version" in node.knobs()


def range_from_input(node):
    aitools_nuke.range_from_input(node, TITLE)


def add_backend_knobs(node):
    if node.knob("backend_state") is not None:
        return
    node.addKnob(nuke.Text_Knob("backend_state", "backend", tr("press Check backend")))
    for action, label in (("start", "Start backend"), ("stop", "Stop backend"), ("status", "Check backend")):
        button = nuke.PyScript_Knob(
            "backend_" + action, tr(label),
            "import alphagen_nuke; alphagen_nuke.control_backend(nuke.thisNode(), {!r})".format(action),
        )
        button.setFlag(nuke.STARTLINE) if action == "start" else button.clearFlag(nuke.STARTLINE)
        node.addKnob(button)


def control_backend(node, action):
    # No source needed to control the backend, so not check_can_compute.
    if node.fullName() in aitools_nuke.running:
        nuke.message(tr("{}: {} is already computing.", TITLE, node.fullName()))
        return None
    python = aitools_nuke.engine_python(PROJECT_ROOT, "ALPHAGEN_PYTHON")
    if not os.path.isfile(python):
        nuke.message(tr("{} engine not installed in:\n{}\nRun the tool's installer first.", TITLE, PROJECT_ROOT))
        return None
    if action == "stop" and not nuke.ask(tr("Stop the shared ComfyUI backend? It will only stop if its queue is idle.")):
        return None

    def on_event(_job, event):
        if event["event"] == "backend":
            aitools_nuke.on_main_thread(node["backend_state"].setValue, event["message"])

    def finished(_job, _node):
        return _node["backend_state"].value()

    job = aitools_nuke.EngineJob(
        node, [python, "-m", "alphagen", "--backend", action, "--json-progress"], PROJECT_ROOT, TITLE,
        verb=tr("backend"), on_event=on_event, finished=finished,
    )
    return job.start_on(node)


def _engine(node):
    python = aitools_nuke.check_can_compute(node, TITLE, PROJECT_ROOT, "ALPHAGEN_PYTHON")
    if python is None:
        return None
    try:
        result = subprocess.run(
            [python, "-m", "alphagen", "--check"],
            cwd=PROJECT_ROOT, env=aitools_nuke.engine_env(),
            stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
            universal_newlines=True, timeout=75,
        )
    except Exception as exc:
        nuke.message(tr("{}: runtime check failed: {}", TITLE, exc))
        return None
    if result.returncode:
        nuke.message(tr("{} is not ready:\n{}", TITLE, result.stdout.strip()))
        return None
    return python


def build_command(
    python, input_pattern, output_pattern, first, last, fps, resolution, seed, layer,
    force=False, subjects=None, low_vram=False, id_map=None,
):
    command = [
        python, "-m", "alphagen",
        "--input", input_pattern,
        "--output", output_pattern,
        "--first", str(first),
        "--last", str(last),
        "--fps", str(fps),
        "--resolution", str(resolution),
        "--seed", str(seed),
        "--layer", layer,
        "--json-progress",
    ]
    if subjects:
        command.extend(["--subjects-json", json.dumps(subjects, separators=(",", ":"))])
    if id_map:
        command.extend(["--id-map", id_map["pattern"], "--ids", ",".join(str(i) for i in id_map["ids"])])
    if low_vram:
        command.append("--low-vram")
    if force:
        command.append("--force")
    return command


def _subjects_for_compute(node, first, last, width, height):
    """Convert active Nuke Viewer points to plate pixels with a top-left origin."""
    subjects = []
    for number in _subject_numbers(node):
        if not node["subject_enabled_{}".format(number)].value():
            continue
        frame = int(node["subject_frame_{}".format(number)].value())
        if not first <= frame <= last:
            raise ValueError(tr("subject {} is on frame {}, outside the range {}-{}", number, frame, first, last))
        x, y = node["subject_point_{}".format(number)].getValue()
        x, y = float(x) - 0.5, float(height) - float(y) - 0.5
        if not (0.0 <= x < width and 0.0 <= y < height):
            raise ValueError(tr("subject {} is outside the plate bounds ({}x{})", number, width, height))
        exclude = node.knob("subject_exclude_{}".format(number))
        include = 0 if exclude is not None and exclude.value() else 1
        subjects.append([frame, x, y, _subject_id(node, number), include])
    return subjects


def _split_points(subjects, hits, everyone=()):
    """(SAM prompts, map people) from the panel points: points on a person of the map pick it.

    Only exclude points on people (no include anywhere) mean ``everyone`` but them.
    """
    included = {hit for point, hit in zip(subjects, hits) if hit and point[4]}
    dropped = {hit for point, hit in zip(subjects, hits) if hit and not point[4]}
    clicks = [point for point, hit in zip(subjects, hits) if not hit]
    if dropped and not included and not any(point[4] for point in clicks):
        included = set(everyone)
    # An exclude off the map only refines SAM clicks of its own subject.
    with_includes = {point[3] for point in clicks if point[4]}
    clicks = [point for point in clicks if point[3] in with_includes]
    return clicks, sorted(included - dropped)


def _sam_points(node):
    """The SAM part of the selection, to tell whether the red preview still matches it."""
    first, last = int(node["range_first"].value()), int(node["range_last"].value())
    source = node.input(0)
    image_format = source.format() if source is not None else nuke.root().format()
    try:
        subjects = _subjects_for_compute(node, first, last, image_format.width(), image_format.height())
    except ValueError:
        return None
    return _split_points(subjects, [_hit(node, number) for number in _active_numbers(node)])[0]


def _describe_setup(event):
    return tr("{}x{}, {} px short edge, {} chunks", event["width"], event["height"], event["size"], event["chunks"])


def _finished(job, node):
    return tr("generated {} alpha frames in {:.1f}s", job.result["written"], job.result["seconds"])


def _source_pattern(node, cache_dir, first, last):
    input_pattern = aitools_nuke.direct_read(node)
    if input_pattern is not None:
        return input_pattern, False
    return aitools_nuke.render_for_compute(
        node, TITLE, "Source", "rgb", [(first, last)],
        os.path.join(cache_dir, "source"), "source",
    ), True


def _range_and_subjects(node):
    """(first, last, subjects) from the panel, or None after telling the user what is wrong."""
    if node.input(0) is None:
        nuke.message(tr("{}: connect a source first.", TITLE))
        return None
    first = int(node["range_first"].value())
    last = int(node["range_last"].value())
    if last < first:
        nuke.message(tr("{}: last frame is before first frame.", TITLE))
        return None
    image_format = node.input(0).format()
    try:
        subjects = _subjects_for_compute(node, first, last, image_format.width(), image_format.height())
    except ValueError as exc:
        nuke.message(tr("{}: {}", TITLE, exc))
        return None
    return first, last, subjects


def _preview_finished(job, node):
    aitools_nuke.reload_cache(node, "SubjectPreview")
    node["show_preview"].setValue(True)
    if _has_people_map(node):
        refresh_selection(node)
    return tr("selection preview: {} frames in {:.1f}s (red); press Compute when it looks right",
              job.result["written"], job.result["seconds"])


def preview(node):
    """Track only the SAM selection over the range and show it tinted red."""
    if "show_preview" not in node.knobs():
        nuke.message(tr("{}: this node predates the selection preview; create a new Alpha Gen node.", TITLE))
        return None
    values = _range_and_subjects(node)
    if values is None:
        return None
    first, last, subjects = values
    if not subjects:
        nuke.message(tr("{}: add a point on the subject first (+ point).", TITLE))
        return None
    if _has_people_map(node):
        subjects = _split_points(subjects, [_hit(node, number) for number in _active_numbers(node)])[0]
        if not subjects:
            nuke.message(tr("{}: every point is on an analyzed person, whose selection already shows; "
                            "Preview selection is only needed for points off the people.", TITLE))
            return None
        node["preview_subjects_json"].setValue(json.dumps(subjects))
    python = aitools_nuke.check_can_compute(node, TITLE, PROJECT_ROOT, "ALPHAGEN_PYTHON")
    if python is None:
        return None
    cache_dir = os.path.dirname(aitools_nuke.cache_pattern(node, "alphagen"))
    input_pattern, rendered = _source_pattern(node, cache_dir, first, last)
    if input_pattern is None:
        return None
    pattern = os.path.join(cache_dir + "_preview", "subject.####.exr").replace("\\", "/")
    node["preview_pattern"].setValue(pattern)
    command = [
        python, "-m", "alphagen", "--preview-subjects",
        "--input", input_pattern, "--output", pattern,
        "--first", str(first), "--last", str(last),
        "--subjects-json", json.dumps(subjects, separators=(",", ":")),
        "--json-progress",
    ]
    job = aitools_nuke.EngineJob(
        node, command, PROJECT_ROOT, TITLE, verb=tr("previewing selection"),
        rendered=[os.path.dirname(input_pattern)] if rendered else [],
        finished=_preview_finished,
    )
    return job.start_on(node)


def _analysis_finished(job, node):
    aitools_nuke.reload_cache(node, "PeopleMap")
    for number in _active_numbers(node):
        _update_hit(node, number)
    refresh_selection(node)
    node["show_preview"].setValue(True)
    manifest = os.path.join(os.path.dirname(node["analysis_pattern"].value()), "analysis.json")
    try:
        with open(manifest) as handle:
            people = json.load(handle)["people"]
    except (OSError, ValueError, KeyError):
        people = "?"
    return tr("{} people found and tracked in {:.1f}s; drag a point onto the one you want", people,
              job.result["seconds"])


def analyze(node):
    """Find everyone on the current frame and track them over the range into the person map."""
    if not _has_people_map(node):
        nuke.message(tr("{}: this node predates the person map; create a new Alpha Gen node.", TITLE))
        return None
    values = _range_and_subjects(node)
    if values is None:
        return None
    first, last, _subjects = values
    key = min(max(int(nuke.frame()), first), last)
    python = aitools_nuke.check_can_compute(node, TITLE, PROJECT_ROOT, "ALPHAGEN_PYTHON")
    if python is None:
        return None
    cache_dir = os.path.dirname(aitools_nuke.cache_pattern(node, "alphagen"))
    input_pattern, rendered = _source_pattern(node, cache_dir, first, last)
    if input_pattern is None:
        return None
    pattern = os.path.join(cache_dir + "_people", "id.####.exr").replace("\\", "/")
    node["analysis_pattern"].setValue(pattern)
    command = [
        python, "-m", "alphagen", "--analyze-subjects", str(key),
        "--input", input_pattern, "--output", pattern,
        "--first", str(first), "--last", str(last), "--json-progress",
    ]
    job = aitools_nuke.EngineJob(
        node, command, PROJECT_ROOT, TITLE, verb=tr("analyzing people"),
        rendered=[os.path.dirname(input_pattern)] if rendered else [],
        finished=_analysis_finished,
    )
    return job.start_on(node)


def compute(node, force=False, wait=False):
    values = _range_and_subjects(node)
    if values is None:
        return None
    first, last, subjects = values
    id_map = None
    if _has_people_map(node):
        had_points = bool(subjects)
        subjects, people = _split_points(
            subjects, [_hit(node, number) for number in _active_numbers(node)], _everyone(node),
        )
        if people:
            id_map = {"pattern": node["analysis_pattern"].value(), "ids": people}
        if had_points and not subjects and not people:
            nuke.message(tr("{}: no subject is selected; add an include point or reset points for automatic selection.", TITLE))
            return None

    python = _engine(node)
    if python is None:
        return None

    cache_pattern = aitools_nuke.cache_pattern(node, "alphagen")
    cache_dir = os.path.dirname(cache_pattern)
    input_pattern, rendered = _source_pattern(node, cache_dir, first, last)
    if input_pattern is None:
        return None

    layer = output_layer(node)
    register_layer(node)
    for name, target in (("CopyAlpha", "rgba.alpha"), ("CopyLayer", layer)):
        copy = node.node(name)
        if copy is not None:
            copy["from0"].setValue(layer)
            copy["to0"].setValue(target)
    cache_layer = node.node("CacheLayer")
    if cache_layer is not None:
        cache_layer["channels"].setValue(layer)
    resolution = RESOLUTIONS[int(node["resolution"].getValue())][1]
    command = build_command(
        python, input_pattern, cache_pattern, first, last,
        float(nuke.root()["fps"].value()), resolution,
        int(node["seed"].value()), layer, force or rendered, subjects=subjects,
        low_vram=bool(node["low_vram"].value()) if "low_vram" in node.knobs() else False,
        id_map=id_map,
    )
    job = aitools_nuke.EngineJob(
        node, command, PROJECT_ROOT, TITLE, verb=tr("matting"),
        rendered=[os.path.dirname(input_pattern)] if rendered else [],
        describe_setup=_describe_setup, finished=_finished,
    )
    return job.start_on(node, wait)
