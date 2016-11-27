-- Safer than image.scale because it won't introduce pixel artifacts

function drawBig(inp, p)
    local inp = inp:clone()

    local inp_size = inp:size()
    local inp_h, inp_w
    if inp_size:size(1) == 3 then
        inp_h = inp_size[2]
        inp_w = inp_size[3]
    else
        -- Replicate into a 3 channel grayscale image
        inp_h = inp_size[1]
        inp_w = inp_size[2]
        inp = inp:view(1, inp_h, inp_w)
        inp = inp:repeatTensor(3, 1, 1)
    end

    local img = torch.zeros(3, inp_h * p, inp_w * p)

    for y = 1, inp_h do
        for x = 1, inp_w do
            for c = 1, 3 do
                i = inp[c][y][x]
                img[{c, {(y - 1) * p + 1, y * p}, {(x - 1) * p + 1, x * p}}] = i
            end
        end
    end
    return img
end

