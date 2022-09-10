import envs

if __name__ == '__main__':
    print('Guess in the following format:')
    print('x y value')
    print('Where x and y are the coorindates on the grid')
    print('and value is the value to place there.')
    env = envs.SudokuEnv1(mask_rate=0.1)
    env.reset()
    done = False
    while not done:
        env.print_render()
        i = input('guess > ')
        if len(i.split(' ')) != 3:
            print('Incorrect action format')
        (x, y, value) = i.split(' ')
        try:
            x = int(x)
            y = int(y)
            value = int(value) - 1
        except e:
            print('Invalid action value(s) -- must be integer')

        if x < 0 or x > 8 or y < 0 or y > 8 or value < 0 or value > 8:
            print('Invalid action value(s) -- must be {0,8} for x,y and {1,9} for value')
            continue
        _, reward, done, _ = env.step((x, y, value))
    if reward > 0:
        print('You won!')
    else:
        print(f'You lost!')
        env.print_render(env.solution)
        