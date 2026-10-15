# Learning From Shoddy Human Demonstration

<p align="center">
	<a href="#tldr">TL;DR</a> &nbsp;|&nbsp;
	<a href="#usage">Usage</a> &nbsp;|&nbsp;
	<a href="#monkey-see-robot-do">Story</a> &nbsp;|&nbsp;
	<a href="#reference">References</a>
</p>

> [!TIP]
> It is often said, “Imitation is the sincerest form of flattery that mediocrity can pay to greatness.” But in the field of robotics, imitation is true greatness.

<table border="0">
	<tr><td align="center">
		<img src="docs/task.png" alt="Image of the simulated task" width="400"><br>
		<sub>Fig. 1: LIBERO Goal task 8: move the bowl onto the plate.</sub>
	</td></tr>
</table>

## TL;DR

- **Goal:** Teach a simulated robot (LIBERO Goal task 8: put the bowl on the plate) from a single, shoddy phone video of a human, with no teleoperated demonstrations.
- **Visual similarity reward:** Frames from the robot's wrist camera are compared with the human video frames in a pre-trained encoder's embedding space (DINOv3/EUPE). Cosine similarity is the reward. This alone gets the robot moving towards the bowl and to the grasping point.
- **Extra supervision:** SAM 2 tracks the fingers and a strip of tape in the human video. This gives a gripper open/close signal and a camera-height signal. With these, the robot grasps the bowl.
- **Final hurdle:** The engineered rewards stop tracking task success, so I lowered the learning rate, raised the entropy and switched to a sparse success reward. The final policy reaches over 40% success.
- **Caveats:** The policy exploits the lack of penalties (violent, odd-angle grasps). Brute-force RL with only a binary reward did not get anywhere in 1M steps.
- **Jump to:** [Usage](#usage) to run it, or keep reading for the [Story](#monkey-see-robot-do).

## Usage

Create the venv:

```sh
uv venv --python 3.10
source .venv/bin/activate 
uv pip install -r requirements.txt
export PYTHONPATH=src:$PYTHONPATH # make sure files in src gets picked up
```

Train a policy:
```sh
uv run ./scripts/train.py \
			--human-demo-embeddings data/human_demo_embeddings.npy \
			--total-timesteps 1000000 \
			--eval-freq 200000 \
			--ckpt-freq 200000 \
			--similarity-weight 0.0 \
			--success-weight 500.0 \
			--gripper-weight 0.0 \
			--z-weight 0.0 \
			--ent-coef 0.01
```

To compile a human demo video into the embeddings file, first split the video into numbered JPEG frames with FFmpeg. The folder must be named after the video, with `_frames` appended:
```sh
mkdir -p data/human_demo_frames
ffmpeg -i data/human_demo.mp4 -q:v 2 -start_number 1 data/human_demo_frames/%06d.jpg
```

Download SAM 2.1 checkpoint. Run from the repo root. The file is about 900 MB.
```bash
wget -nc -P sam2_checkpoints https://dl.fbaipublicfiles.com/segment_anything_2/092824/sam2.1_hiera_large.pt
```

Then track the fingers and tabletop tape with SAM 2 to write `gripper_and_object_scale.csv` into that folder:
```sh
uv run scripts/extract_supervision_csv.py \
		--frames-dir data/human_demo_frames \
		--tape-points 710 100 710 200 710 250 \
		--finger-points 200 600 800 600
```
The `--tape-points` and `--finger-points` are `x y` clicks on frame 000001 (foreground prompts for SAM 2). Adjust them to your own video.

Finally, encode the video:
```sh
uv run scripts/encode_human_video.py \
		--video data/human_demo.mp4 \
		--output data/human_demo_embeddings.npy \
		--gripper-closed-threshold=200
```

## Monkey See, Robot Do

Fig. 1 is an image of task 8 in the Libero goal benchmark suite[^1]. The robot is asked to pick up the bowl and place it on the plate. By learning from 50 episodes of teleoperated demonstrations, numerous papers have reported close to 100\% success rate on the Libero Task just from pure behaviour cloning[^2][^3][^4]. Teleoperated data have zero visual gap and morphology gap, hence establishing the translation from current camera image to next robot action is relatively easy to do.

AlphaNIST[^5] also showed very recently that massively parallel simulation can brute force the emergence of very dexterous multistep skills. Hell, GPT-Astra could probably zero-shot this bad boy if I'm willing to pay the cloud thinking machine some of my hard-earned cash[^8]. Alas, I'm not.

Acknowledging that some of these much better approaches exist, I want to explore an alternative route. This is a video I took on my phone, as the repo title suggests, shoddily produced:

<table border="0">
	<tr><td align="center">
		<a href="data/human_demo_4.mp4"><img src="docs/human_demo_4.gif" alt="Human demonstration" width="400"></a><br>
		<sub>Fig. 2: Phone-recorded human demonstration.</sub>
	</td></tr>
</table>

Any human who looks at this video will immediately say, "Ah, I know what to do." even though the appearance of the props are completely different in the simulation. That means this short video already contains all the necessary information to tell someone how to complete the task. Now, the hard problem is: how do we extract this information and give it to the robot in a way that it understands?

### Pre-trained visual encoders

Foundational pre-trained visual encoders are trained on internet-scale images to embed semantic understanding into a latent space. DINOv3 show superior depth and segmentation capabilities[^6]. EUPE Distils that information into a more compact model[^7]. 

I wonder if it is possible to leverage this embedding space to transfer some of the knowledge from the human video to the robot. If I calculate the similarity of the robot's wrist camera with frames from the human video, I can use this as a reward to bootstrap the RL.

Embed both the robot wrist image $z_r(j)$ and the human video frame $z_h(t)$ into embeddings $E_r$ and $E_h$. Then the similarity is just the cosine of the embedded vectors

$$
\mathrm{sim}(\mathbf{E}_R,\mathbf{E}_H)=
\frac{\mathbf{E}_R \cdot \mathbf{E}_H}
{\|\mathbf{E}_R\|_2\,\|\mathbf{E}_H\|_2}
$$

The RL converged within 60k steps to this:

<table border="0">
	<tr><td align="center">
		<a href="docs/visual_similarity_only.mp4"><img src="docs/visual_similarity_only.gif" alt="Visual similarity run" width="400" height="200"></a><br>
		<sub>Fig. 3: Robot evaluation after visual-similarity training.</sub>
	</td>
    <td align="center">
		<img src="docs/pure_visual_similarity_train_graph.png" alt="Pure visual similarity training graph" width="206" height="200"><br>
		<sub>Fig. 4: Training curve (average reward vs simulation steps) for the visual-similarity run.</sub>
	</td></tr>
</table>

I'm blown away by how structurally similar it is already to the original video. The robot, like the human video, goes down towards the bowl and even approaches the bowl on the right-hand side. This is based purely on comparing the direction of the high-dimensional vectors in the embedding space.

### More Supervisions

But the visual similarity alone wasn't able to carry the whole task to completion. Because once the robot picks up the bowl, almost the entirety of the camera view is filled up by the bowl itself. Then, from that point on, all the frames look similar.

I engineered two more rewards to try to guide the robots into completing the task. 

<table border="0">
	<tr><td align="center">
		<img src="docs/segment_anything.png" alt="Segment anything" width="400"><br>
		<sub>Fig. 5: Prompt points in frame zero of the video for SAM 2 tracking of the fingers and tabletop tape.</sub>
	</td></tr>
</table>

By using the Segment Anything Model 2 (SAM 2), I track the location of the fingers and the black tape on the table in the original human video. This is not as generalizable as pure vision embedding similarity. But all I had to manually provide was a few prompt points on the very first frame of the video.

The tracked fingers are used to calculate a gripper open/close state. And the width of the tape on the tabletop gives me an indication of how high the camera is. This is turned into two supervision signals. One guides the gripper to close. The other guides the robot to go down-up-down with its motion.

<table border="0">
	<tr></tr>
</table>

<table border="0">
	<tr><td align="bottom">
		<img src="docs/z_supervision.png" alt="Object-height supervision" height="200"><br>
		<sub>Fig.6: Object-height supervision from the tabletop tape.</sub>
	</td>
    <td align="bottom">
		<img src="docs/gripper_supervision.png" alt="Gripper supervision" height="200"><br>
		<sub>Fig.7: Gripper-state supervision from the human demonstration.</sub>
	</td>
    </tr>
</table>

Using these additional rewards, I get the following result:

The policy now successfully grabs the bowl and makes a vague attempt to lift it.

<table border="0">
	<tr><td align="center">
		<a href="docs/all_rewards.mp4"><img src="docs/all_rewards.gif" alt="Combined rewards run" height="200"></a><br>
		<sub>Fig. 8: Robot evaluation with the combined reward signals.</sub>
	</td><td align="center">
		<img src="docs/all_rewards_training_graph.png" alt="Combined rewards training graph" height="200"><br>
		<sub>Fig. 9: Training curve, reward vs steps, for the combined-reward run.</sub>
	</td></tr>
</table>


### The final hurdle

At this point, it seems that the engineered rewards no longer align with task success rate. More training or higher rewards does not improve success. Success rate sits still at zero.

This is the point where we should let go and let the robot explore the world and find its own success. In the final leg of training, the learning rate is turned down so that it does not deviate too much from the current policy. The entropy is increased to encourage exploration. Then, finally, once the policy is able to maintain a good level of success rate, all other rewards are turned off, so the policy can align itself to get the best success rate.


<table border="0">
	<tr><td align="center">
		<img src="docs/success_rate_final_hurdle.png" alt="Average success rate across 2 runs. First run in red with all reward signals, 2nd run in orange continuing from the first using only sparse success reward." width="200"><br>
		<sub>Fig. 10: Average success rate across 2 runs.<br>
			First run in red with all reward signals.<br>
			Second run in orange continuing from the first using only sparse success reward.</sub>
	</td>
    </tr>
</table>

<table border="0">
	<tr>
		<td align="center">
		<a href="docs/final_hurdle.mp4"><img src="docs/final_hurdle.gif" alt="Final-hurdle robot evaluation" width="400" height="200"></a><br>
		<sub>Fig. 11: Final robot policy. Success rate > 40%</sub>
		</td>
	</tr>
</table>

### Reward Hacking

You can see in the final policy that the robot picks up the bowl at rather an odd angle and very violently. This is clearly not desirable in a real-world scenario. But in the simulation, there are zero penalties discouraging such behaviour. This exhibits one of the key downsides of reinforcement learning. This is why human teleoperated expert trajectory Can be a very valuable resource to teach the robot a desirable behaviour That may not be easily specified by a reward.

### Why not brute force

Some people may ask, "Why not just brute force RL the whole thing?". This is 1 million steps of RL with binary success reward.

<table border="0">
	<tr>
		<td align="center">
		<a href="docs/binary_reward.mp4"><img src="docs/binary_reward.gif" alt="Binary reward robot evaluation" width="400" height="200"></a><br>
		<sub>Fig. 12: Robot evaluation with the binary success reward.</sub>
		</td>
	</tr>
</table>

AlphaNIST showed that it is possible, but it requires far more compute than an individual can reasonably be expected to have. In fact, the paper found that dense reward guidance actually hurts the final success rate. This is a very interesting line of research for robotics. If the cost of simulation keeps falling and the sim-to-real gap can be reasonably closed, mass-scale sparse-reward RL might be a good approach for unlocking dexterous skills in robots, especially those whose morphologies differ from humans' and cannot reasonably be teleoperated.

## References

[^1]: Liu, Bo, Yifeng Zhu, Chongkai Gao, Yihao Feng, Qiang Liu, Yuke Zhu, and Peter Stone. "Libero: Benchmarking knowledge transfer for lifelong robot learning." Advances in Neural Information Processing Systems 36 (2023): 44776-44791. https://libero-project.github.io/main.html
[^2]: Shukor, Mustafa, Dana Aubakirova, Francesco Capuano, Pepijn Kooijmans, Steven Palma, Adil Zouitine, Michel Aractingi et al. "Smolvla: A vision-language-action model for affordable and efficient robotics." arXiv preprint arXiv:2506.01844 (2025).
[^3]: Shi, Hao, Bin Xie, Yingfei Liu, Lin Sun, Fengrong Liu, Tiancai Wang, Erjin Zhou, Haoqiang Fan, Xiangyu Zhang, and Gao Huang. "Memoryvla: Perceptual-cognitive memory in vision-language-action models for robotic manipulation." In International Conference on Learning Representations, vol. 2026, pp. 18567-18602. 2026.
[^4]: Black, Kevin, Noah Brown, Danny Driess, Adnan Esmail, Michael Equi, Chelsea Finn, Niccolo Fusai et al. "$\pi_0 $: A Vision-Language-Action Flow Model for General Robot Control." arXiv preprint arXiv:2410.24164 (2024).
[^5]: Zhang, Octi, Iretiayo Akinola, Rosario Scalise, Mateo Guaman Castro, Patrick Yin, Tobias Widmer, Miles Macklin et al. "AlphaNIST: Emergent Assembly Skills from Binary Task Rewards." arXiv preprint arXiv:2610.13354 (2026). https://alpha-nist.github.io/
[^6]: Siméoni, Oriane, Huy V. Vo, Maximilian Seitzer, Federico Baldassarre, Maxime Oquab, Cijo Jose, Vasil Khalidov et al. "Dinov3." arXiv preprint arXiv:2508.10104 (2025). https://github.com/facebookresearch/dinov3
[^7]: Zhu, Chenchen, Saksham Suri, Cijo Jose, Maxime Oquab, Marc Szafraniec, Wei Wen, Yunyang Xiong et al. "Efficient Universal Perception Encoder." arXiv preprint arXiv:2603.22387 (2026). https://github.com/facebookresearch/EUPE
[^8]: Zhang, Wenbo, Kaixuan Wang, Yutao Ouyang, Xiaoyu Huang, Liyang Li, Kailun Su, Weiyang Jin et al. "An Unexpected Robot Policy: Early Evaluations of GPT-6 Astra on RoboDojo and Beyond." arXiv preprint arXiv:2609.24170 (2026).
