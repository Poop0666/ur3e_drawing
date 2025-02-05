import customtkinter as ctk

class Tooltip:
    """Creates a tooltip that appears when hovering over a widget."""
    def __init__(self, widget, text):
        self.widget = widget
        self.text = text
        self.tooltip_window = None
        self.mouse_over_tooltip = False

        # Bind hover events to the widget
        self.widget.bind("<Enter>", self.show_tooltip)
        self.widget.bind("<Leave>", self.check_leave)

    def show_tooltip(self, event=None):
        """Show the tooltip near the widget."""
        if self.tooltip_window:
            return  # Avoid creating multiple tooltips

        # Create the tooltip window
        self.tooltip_window = ctk.CTkToplevel(self.widget)
        self.tooltip_window.overrideredirect(True)  # Remove window borders
        self.tooltip_window.geometry(f"+{self.widget.winfo_rootx() + 20}+{self.widget.winfo_rooty() + 20}")

        # Tooltip Label
        label = ctk.CTkLabel(self.tooltip_window, text=self.text, fg_color="white", text_color="black", corner_radius=5, padx=5, pady=3)
        label.pack()

        # Bind events to the tooltip window to prevent flickering
        self.tooltip_window.bind("<Enter>", self.on_tooltip_hover)
        self.tooltip_window.bind("<Leave>", self.check_leave)

    def on_tooltip_hover(self, event=None):
        """Mark that the mouse is over the tooltip."""
        self.mouse_over_tooltip = True

    def check_leave(self, event=None):
        """Hide the tooltip only if the mouse leaves both the widget and tooltip."""
        if self.tooltip_window:
            self.mouse_over_tooltip = False
            self.widget.after(100, self.hide_tooltip)

    def hide_tooltip(self):
        """Destroy the tooltip window if the mouse is no longer over it."""
        if not self.mouse_over_tooltip and self.tooltip_window:
            self.tooltip_window.destroy()
            self.tooltip_window = None


if __name__== "__main__":
    # Create main window
    root = ctk.CTk()
    root.geometry("300x200")

    # Round info button
    info_button = ctk.CTkButton(
        root,
        text="i",
        width=30,
        height=30,
        fg_color="gray",
        text_color="white",
        corner_radius=15,  # Round shape
    )
    info_button.pack(pady=20)

    # Attach tooltip to the button
    Tooltip(info_button, "This is an information tooltip!")

    # Run the app
    root.mainloop()
