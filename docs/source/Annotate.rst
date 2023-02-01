Annotate
========

``MapReader`` provides a flexible way to set up a new annotation task in three simple steps:

    1. Edit the ``./annotation_tasks.yaml`` file.
       
       This file contains two sections - ``tasks`` and ``paths``. The ``tasks`` section is used to specify a task and its labels. e.g. : 
       
       .. code :: yaml

            # ---------------------------------------
            # Define an annotation task
            # This includes:
            # 1. a name (e.g., building_simple or rail_space, see below)
            # 2. a list of labels to be used for this task
            # ---------------------------------------
            tasks:
              building_simple:
                labels: ["No", "building"]
              rail_space:
                labels: ["No", "rail space"]

