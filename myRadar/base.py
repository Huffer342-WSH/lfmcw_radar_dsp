class Printable:
    """
    A base class that provides a customizable __repr__ method for better object printing.
    """

    def __repr__(self):
        # Default indentation and formatting parameters
        indent = " " * 4
        max_line_length = 80
        max_total_length = 50000
        truncated_marker = "... (truncated due to length)"

        # Gather attributes and their values
        attributes = []
        for name in dir(self):
            if not name.startswith("_") and not callable(getattr(self, name)):
                value = getattr(self, name)
                value_repr = repr(value)
                # Indent multiline values
                if "\n" in value_repr:
                    indented_value = "\n".join(f"{indent}{line}" for line in value_repr.splitlines())
                    value_repr = f"\n{indented_value}"
                attributes.append(f"{indent}{name}={value_repr}")

        # Format the final representation
        class_name = self.__class__.__name__
        repr_body = ",\n".join(attributes)
        full_repr = f"{class_name}(\n{repr_body}\n)"

        # Truncate if exceeds maximum total length
        if len(full_repr) > max_total_length:
            full_repr = f"{full_repr[:max_total_length]}{truncated_marker}"

        return full_repr
