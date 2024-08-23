def track_rm_usage(target_product, raw_material, data, visited=None, depth=0, max_depth=10):
    if visited is None:
        visited = set()

    if target_product in visited or depth > max_depth:
        return 0

    visited.add(target_product)

    total_usage = 0

    # Find all processes that produce the target product
    product_processes = data[(data['product_name'] == target_product) & (data['type'] == 'PRODUCT')]

    for _, process in product_processes.iterrows():
        process_name = process['process']
        process_flow_amount = process['flow_amount']

        # Find all raw materials used in this process
        raw_materials_used = data[(data['process'] == process_name) & (data['product_name'] == raw_material) & (
                    data['type'] == 'RAW MATERIALS')]

        for _, material in raw_materials_used.iterrows():
            # Calculate the usage of the raw material in this process
            material_usage = -material['value'] * process_flow_amount
            total_usage += material_usage

            # Recursively calculate usage for materials contributing to this process
            contributing_materials = data[(data['process'] == process_name) & (data['type'] == 'RAW MATERIALS') & (
                        data['product_name'] != target_product)]
            for _, contrib_material in contributing_materials.iterrows():
                material_name = contrib_material['product_name']
                required_amount = -contrib_material['value'] * process_flow_amount  # Scale by the process output
                if required_amount > 0:
                    material_consumption = track_rm_usage(material_name, raw_material, data, visited,
                                                                        depth + 1, max_depth)
                    total_usage += material_consumption * required_amount

    visited.remove(target_product)
    return total_usage


def calculate_raw_material_requirement(data, target_product, process, raw_material):
    # Function to recursively calculate raw material required for production of target_product
    visited = set()  # To avoid revisiting the same process

    def recursive_requirement(current_product, current_process, coefficient=1):
        # Mark this process as visited to avoid infinite loops
        if (current_product, current_process) in visited:
            return 0
        visited.add((current_product, current_process))

        # Filter data for entries related to the current process
        current_entries = data[data['process'] == current_process]

        # Sum up all raw materials that go into making the current product under its process
        total = 0
        for _, row in current_entries.iterrows():
            if row['type'] == 'RAW MATERIALS' and row['product_name'] == raw_material:
                # Direct use of the raw material
                total += -row['value'] * coefficient
            elif row['type'] == 'RAW MATERIALS':
                # Recursively calculate the input required for each raw material
                total += recursive_requirement(row['product_name'], row['process'], -row['value'] * coefficient)

        return total

    # Start recursion from the target product and specified process
    target_entries = data[
        (data['product_name'] == target_product) & (data['process'] == process) & (data['type'] == 'PRODUCT')]
    if target_entries.empty:
        return 0  # No such product-process combination

    target_entry = target_entries.iloc[0]
    return recursive_requirement(target_product, process, 1 / target_entry['value'])
