# Use the official Firedrake image as the base
FROM firedrakeproject/firedrake:latest

# Set environment variables
ENV SHELL=/usr/bin/zsh
ENV TERM=xterm-256color

# Switch to root user to install system packages
USER root

# Install zsh and other useful tools
RUN apt-get update && apt-get install -y \
    zsh git gmsh \
    dvipng texlive-latex-extra texlive-fonts-recommended cm-super \
    libxcb-xinerama0-dev libxcb-xkb-dev libxcb-shape0-dev \
    libxcb-render-util0-dev libxcb-randr0-dev libxcb-image0-dev \
    libxcb-icccm4-dev libxcb-util-dev libxcb-keysyms1-dev \
    && rm -rf /var/lib/apt/lists/*

# Install Oh My Zsh
RUN sh -c "$(curl -fsSL https://raw.githubusercontent.com/ohmyzsh/ohmyzsh/master/tools/install.sh)" "" --unattended

# Clone zsh plugins
RUN git clone https://github.com/zsh-users/zsh-autosuggestions ~/.oh-my-zsh/custom/plugins/zsh-autosuggestions
# RUN git clone https://github.com/zsh-users/zsh-syntax-highlighting.git ~/.oh-my-zsh/custom/plugins/zsh-syntax-highlighting

# Copy your local .zshrc file into the container
COPY .zshrc /home/firedrake/.zshrc

# Fix permissions for the .zshrc file
RUN chown firedrake:firedrake /home/firedrake/.zshrc

# Set zsh as the default shell for the firedrake user
RUN chsh -s /usr/bin/zsh firedrake

# Set the working directory
WORKDIR /home/firedrake/shared

# Switch back to the firedrake user
USER firedrake

# Activate the Firedrake virtual environment by default
RUN echo "source ~/firedrake/bin/activate" >> ~/.zshrc

# Install additional Python packages
RUN source ~/firedrake/bin/activate && \
    pip install --no-cache-dir --break-system-packages pdbpp PyQt5 tikzplotlib siphash24 pyvista tqdm

# # Clone defcon repository
RUN git clone https://bitbucket.org/pefarrell/defcon.git ~/defcon

# Set up Firedrake cache
RUN mkdir -p ~/.cache/tsfc/

# Start zsh when the container runs
CMD ["zsh"]