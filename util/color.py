def SHOW(s, color=4):
    s = str(s)
    print("\033[3%dm%s\033[0m"%(color, s))

def INFO(s, color=2):
    s = str(s)
    print("\033[3%dm%s\033[0m"%(color, s))

def ERR(s, color=1):
    s = str(s)
    print("\033[3%d;4;1m%s\033[0m"%(color, s))


def DEBUG(s, color=3):
    s = str(s)
    print("\033[3%d;4;1m%s\033[0m"%(color, s))


if __name__ == "__main__":
    SHOW("asdf")
    INFO("asdf")
    ERR("asdf")
    DEBUG("asdf")
    print("\033[31;4m", end="")
    print("Red Underline Text\033[0m\n")
