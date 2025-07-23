FROM ubuntu

USER root
RUN apt update
RUN apt install python3.12 -y

# Necessary Python libraries
RUN apt install python3-sklearn -y
RUN apt install python3-pandas -y
RUN apt install python3-numpy -y

RUN if id ubuntu >/dev/null 2>&1; then userdel -f -r ubuntu; fi && \
    useradd -s /bin/bash -m hacker && \
    passwd -d hacker && \
    ln -s /etc/bash.bashrc /etc/bashrc

# Add exec-suid to the image (important for Python challenges to be able to read the flag):
ADD --chown=0:0 http://github.com/pwncollege/exec-suid/releases/latest/download/exec-suid /usr/bin/exec-suid
RUN chmod 6755 /usr/bin/exec-suid
 