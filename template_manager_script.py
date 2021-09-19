while True:
    preview = node.io['preview'].get()
    node.io['manip'].send(preview)
    node.io['host'].send(preview)
    ${_TRACE} ("Manager took frame from preview and sent it to manip and to host")


